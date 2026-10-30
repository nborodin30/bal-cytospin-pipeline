import os
import gc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import torchvision.transforms.functional as TF
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
import timm
from timm.utils import ModelEmaV2
from PIL import Image
from tqdm import tqdm

# ==========================================
# 1. ПАРАМЕТРЫ И КОНФИГУРАЦИЯ
# ==========================================
BASE_DIR = '/vol/firstbatch'
OUTPUT_DIR = os.path.join(BASE_DIR, 'production_model')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_ARCH = "hf_hub:1aurent/vit_base_patch14_224.dinobloom"
IMG_SIZE = 224
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4  # Эффективный размер батча = 32
EPOCHS = 35
RANDOM_STATE = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TARGET_CLASSES = [
    'Macrophage', 'Macrophage_with_particles', 'Neutrophil', 
    'Erythrocyte', 'Lymphocyte', 'Monocyte', 'Eosinophil', 'Junk'
]
class_to_idx = {name: idx for idx, name in enumerate(TARGET_CLASSES)}
idx_to_class = {idx: name for name, idx in class_to_idx.items()}

# ==========================================
# 2. СБОРКА ДАННЫХ И ПРИВЯЗКА SLIDE_ID
# ==========================================
def map_to_eight_classes(row):
    cls_str = str(row['clean_class']).strip().lower()
    txt = str(row['text_evidence']).lower()
    if any(k in cls_str for k in ['junk', 'debris', 'artifact', 'uncertain']):
        return 'Junk'
    if 'macrophage' in cls_str:
        if 'particle' in cls_str or 'particle' in txt or 'nanotubes' in txt:
            return 'Macrophage_with_particles'
        return 'Macrophage'
    for target in TARGET_CLASSES:
        if target.lower() in cls_str:
            return target
    return None

def parse_slide(path):
    fn = os.path.basename(str(path))
    return fn.split('_cell_')[0] if '_cell_' in fn else 'unknown_slide'

print(f"⏳ Загрузка разметки всех 4 раундов...")
df1 = pd.read_csv(os.path.join(BASE_DIR, 'mapped_annotations_round1.csv'))
df2 = pd.read_csv(os.path.join(BASE_DIR, 'mapped_annotations_round2.csv'))
df3 = pd.read_csv(os.path.join(BASE_DIR, 'mapped_annotations_round3.csv'))
df4 = pd.read_csv(os.path.join(BASE_DIR, 'mapped_annotations_round4.csv'))

p1 = pd.DataFrame({
    'img_path': df1['original_path'],
    'clean_class': df1['clean_class'],
    'text_evidence': df1.get('notes', '').fillna('').astype(str) + " " + df1.get('morphology_details', '').fillna('').astype(str),
    'slide_id': df1['original_path'].apply(parse_slide)
})
p2 = pd.DataFrame({
    'img_path': df2['original_file_path'],
    'clean_class': df2['clean_class'],
    'text_evidence': df2.get('notes', '').fillna('').astype(str),
    'slide_id': df2['original_file_path'].apply(parse_slide)
})
p3 = pd.DataFrame({
    'img_path': df3['original_path'],
    'clean_class': df3['clean_class'],
    'text_evidence': df3.get('notes', '').fillna('').astype(str) + " " + df3.get('morphology_details', '').fillna('').astype(str),
    'slide_id': df3['original_path'].apply(parse_slide)
})
p4 = pd.DataFrame({
    'img_path': 'Round4_SmartMiner/crops/' + df4['original_path'].astype(str),
    'clean_class': df4['clean_class'],
    'text_evidence': '',
    'slide_id': df4['original_slide'].fillna('unknown_slide').astype(str)
})

full_df = pd.concat([p1, p2, p3, p4], ignore_index=True)
full_df['target_class'] = full_df.apply(map_to_eight_classes, axis=1)
full_df = full_df.dropna(subset=['target_class', 'img_path']).reset_index(drop=True)

bad_file = os.path.join(BASE_DIR, 'suspect_labels.csv')
if os.path.exists(bad_file):
    bad_df = pd.read_csv(bad_file)
    init_cnt = len(full_df)
    full_df = full_df[~full_df['img_path'].isin(bad_df['img_path'].tolist())].reset_index(drop=True)
    print(f"🗑️ Удалено токсичных изображений: {init_cnt - len(full_df)}")

full_df['label_idx'] = full_df['target_class'].map(class_to_idx)

# ==========================================
# 3. ПОЛУЧЕНИЕ TRAIN И ЧЕСТНОГО HOLDOUT TEST
# ==========================================
holdout_candidates = [
    os.path.join(BASE_DIR, 'reports_honest_base', 'holdout_manifest.csv'),
    os.path.join(BASE_DIR, 'reports_dinobloom_small', 'holdout_manifest.csv')
]

holdout_path = next((p for p in holdout_candidates if os.path.exists(p)), None)

if holdout_path:
    print(f"📦 Использование существующего манифеста Hold-out: {holdout_path}")
    test_df = pd.read_csv(holdout_path)
    test_paths = set(test_df['img_path'].tolist())
    train_df = full_df[~full_df['img_path'].isin(test_paths)].reset_index(drop=True)
else:
    print("⚠️ Манифест не найден, формируется новый сплит 80/20 по слайдам...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sgkf.split(full_df, full_df['label_idx'], groups=full_df['slide_id']))
    train_df = full_df.iloc[train_idx].reset_index(drop=True)
    test_df = full_df.iloc[test_idx].reset_index(drop=True)
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'holdout_manifest.csv'), index=False)

train_slides = set(train_df['slide_id'].unique())
test_slides = set(test_df['slide_id'].unique())
assert len(train_slides.intersection(test_slides)) == 0, "КРИТИЧЕСКАЯ ОШИБКА: Пересечение слайдов Train и Test!"

print("\n" + "="*60)
print(f"🔹 Обучающая выборка (Train): {len(train_df)} клеток ({len(train_slides)} слайдов)")
print(f"🔹 Контрольный тест (Hold-out): {len(test_df)} клеток ({len(test_slides)} слайдов)")
print("="*60)

# ==========================================
# 4. DATASET И ЗАГРУЗЧИКИ
# ==========================================
class BALDataset(Dataset):
    def __init__(self, dataframe, base_dir, img_size=224, is_train=True):
        self.df = dataframe.reset_index(drop=True)
        self.base_dir = base_dir
        self.target_size = img_size
        
        def smart_pad(img):
            delta_w, delta_h = self.target_size - img.size[0], self.target_size - img.size[1]
            if delta_w <= 0 and delta_h <= 0:
                return img.resize((self.target_size, self.target_size), Image.BILINEAR)
            pad_w, pad_h = max(0, delta_w // 2), max(0, delta_h // 2)
            return TF.pad(img, (pad_w, pad_h, delta_w - pad_w, delta_h - pad_h), padding_mode='reflect')

        base_tf = [
            transforms.Lambda(smart_pad),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation((0, 360)),
                transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.25, hue=0.08),
            ] + base_tf)
        else:
            self.transform = transforms.Compose(base_tf)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_p = os.path.join(self.base_dir, row['img_path']) if not str(row['img_path']).startswith('/') else row['img_path']
        return self.transform(Image.open(img_p).convert('RGB')), int(row['label_idx'])

class_counts = np.bincount(train_df['label_idx'], minlength=len(TARGET_CLASSES))
sampler_weights = torch.from_numpy(1.0 / np.sqrt(class_counts + 1e-6)[train_df['label_idx'].values]).double()
train_sampler = WeightedRandomSampler(weights=sampler_weights, num_samples=len(train_df), replacement=True)

train_loader = DataLoader(BALDataset(train_df, BASE_DIR, IMG_SIZE, is_train=True), batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4, pin_memory=True)
test_loader = DataLoader(BALDataset(test_df, BASE_DIR, IMG_SIZE, is_train=False), batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4)

# ==========================================
# 5. ОБУЧЕНИЕ МОДЕЛИ
# ==========================================
print(f"\n🚀 Инициализация {MODEL_ARCH}...")
model = timm.create_model(
    MODEL_ARCH, pretrained=True, num_classes=len(TARGET_CLASSES), 
    img_size=IMG_SIZE, global_pool='avg', drop_rate=0.20, drop_path_rate=0.10
).to(DEVICE)

num_layers = len(model.blocks)
param_groups = [{'params': model.head.parameters(), 'lr': 1e-4, 'weight_decay': 1e-2}]
for i, block in enumerate(model.blocks):
    param_groups.append({'params': block.parameters(), 'lr': 1e-5 * (0.75 ** (num_layers - 1 - i)), 'weight_decay': 1e-2})
param_groups.append({'params': model.patch_embed.parameters(), 'lr': 1e-5 * (0.75 ** num_layers), 'weight_decay': 1e-2})
param_groups.append({'params': model.norm.parameters(), 'lr': 1e-5, 'weight_decay': 1e-2})

optimizer = torch.optim.AdamW(param_groups)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
model_ema = ModelEmaV2(model, decay=0.997)

print("⚡ Старт обучения финальной продакшен-модели...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f"Ep {epoch+1:02d}/{EPOCHS}", leave=False)
    for step, (images, labels) in enumerate(pbar):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        loss = criterion(model(images), labels) / GRAD_ACCUM_STEPS
        loss.backward()
        
        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            model_ema.update(model)
            
        running_loss += loss.item() * GRAD_ACCUM_STEPS
        
    scheduler.step()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

# Сохранение итогового файла весов для плагина
final_weights_path = os.path.join(OUTPUT_DIR, 'dinobloom_bal_production.pth')
torch.save(model_ema.module.state_dict(), final_weights_path)
print(f"\n💾 Веса для плагина успешно сохранены в:\n   👉 {final_weights_path}")

# ==========================================
# 6. ОЦЕНКА ИТОГОВЫХ МЕТРИК И УВЕРЕННОСТИ
# ==========================================
print("\n" + "="*65)
print("🔬 ТЕСТИРОВАНИЕ НА ИЗОЛИРОВАННОМ HOLD-OUT TEST (4-FOLD TTA)")
print("="*65)

model_ema.module.eval()
all_probs = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Инференс модели"):
        images = images.to(DEVICE)
        outputs = (
            model_ema.module(images) + 
            model_ema.module(torch.rot90(images, 1, [2, 3])) + 
            model_ema.module(torch.rot90(images, 2, [2, 3])) + 
            model_ema.module(torch.rot90(images, 3, [2, 3]))
        ) / 4.0
        
        probs = F.softmax(outputs, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

all_probs = np.concatenate(all_probs, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

preds = np.argmax(all_probs, axis=1)
confidences = np.max(all_probs, axis=1)

accuracy = (preds == all_labels).mean() * 100
macro_f1 = f1_score(all_labels, preds, average='macro')
weighted_f1 = f1_score(all_labels, preds, average='weighted')

print(f"\n🏆 ИТОГОВЫЕ МЕТРИКИ ДЛЯ ПАСПОРТА ПЛАГИНА:")
print(f"• Точность (Test Accuracy):   {accuracy:.2f}%")
print(f"• Macro F1-Score:             {macro_f1:.4f}")
print(f"• Weighted F1-Score:          {weighted_f1:.4f}")
print(f"• Средняя уверенность модели:  {confidences.mean() * 100:.2f}%\n")

# Детальная таблица по классам
rep_dict = classification_report(all_labels, preds, target_names=TARGET_CLASSES, output_dict=True, zero_division=0)
df_rep = pd.DataFrame(rep_dict).transpose().round(4).loc[TARGET_CLASSES]

# Расчет средней уверенности для каждого класса
mean_confs = []
for idx in range(len(TARGET_CLASSES)):
    cls_mask = (preds == idx)
    mean_confs.append(round(float(confidences[cls_mask].mean()), 4) if cls_mask.sum() > 0 else 0.0)

df_rep['mean_confidence'] = mean_confs
print(df_rep.to_string())

# Анализ калибровки уверенности
correct_mask = (preds == all_labels)
print("\n📊 КАЛИБРОВКА УВЕРЕННОСТИ (CONFIDENCE GAP):")
print(f"• Средняя уверенность на ВЕРНЫХ ответах:    {confidences[correct_mask].mean() * 100:.2f}%")
print(f"• Средняя уверенность на ОШИБОЧНЫХ ответах: {confidences[~correct_mask].mean() * 100:.2f}%")

# Сохранение отчета
df_rep.to_csv(os.path.join(OUTPUT_DIR, 'production_model_metrics.csv'))
print(f"\n💾 Полный отчет сохранен в: {os.path.join(OUTPUT_DIR, 'production_model_metrics.csv')}")