import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
import timm
from timm.utils import ModelEmaV2
from PIL import Image
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. КОНФИГУРАЦИЯ
# ==========================================
DATA_DIR = "/vol/firstbatch/manual_cell_exports_1001"
CSV_PATH = os.path.join(DATA_DIR, "annotation_neutrophil_completed.csv")
MODEL_NAME = "hf_hub:1aurent/vit_base_patch14_224.dinobloom"
OUTPUT_WEIGHTS = "/vol/firstbatch/neutrophil_expert.pth"

IMG_SIZE = 224
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4  # Эффективный батч = 32
EPOCHS = 20           # Чуть больше эпох, так как работает EMA и сильная аугментация
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    "Immature_Neu",
    "Mature_Neu_SiglecF+",
    "Mature_Neu_SiglecF-"
]

# ==========================================
# 2. DATASET И УМНЫЙ ПРЕПРОЦЕССИНГ
# ==========================================
class NeutrophilDataset(Dataset):
    def __init__(self, df, base_dir, is_train=True):
        self.df = df.reset_index(drop=True)
        self.base_dir = base_dir
        self.is_train = is_train
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
        
        def smart_pad(img):
            delta_w, delta_h = IMG_SIZE - img.size[0], IMG_SIZE - img.size[1]
            if delta_w <= 0 and delta_h <= 0:
                return img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
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
        img_path = os.path.join(self.base_dir, row["blind_filename"])
        image = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[row["cell_type"]]
        return self.transform(image), label


def train_expert():
    # 1. Загрузка и разбивка данных
    df = pd.read_csv(CSV_PATH)
    train_df, val_df = train_test_split(df, test_size=0.15, stratify=df["cell_type"], random_state=42)
    
    # 2. Взвешенный семплер для баланса классов
    train_labels = [CLASSES.index(cls) for cls in train_df["cell_type"]]
    class_counts = np.bincount(train_labels, minlength=len(CLASSES))
    sampler_weights = torch.from_numpy(1.0 / np.sqrt(class_counts + 1e-6)[train_labels]).double()
    train_sampler = WeightedRandomSampler(weights=sampler_weights, num_samples=len(train_df), replacement=True)

    # 3. Загрузчики данных
    train_dataset = NeutrophilDataset(train_df, DATA_DIR, is_train=True)
    val_dataset = NeutrophilDataset(val_df, DATA_DIR, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4)

    # 4. Инициализация модели (Замораживаем базу, учим только голову)
    print(f"🚀 Инициализация модели-эксперта на базе {MODEL_NAME}...")
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(CLASSES), img_size=IMG_SIZE, global_pool='avg')
    
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

    model = model.to(DEVICE)
    model_ema = ModelEmaV2(model, decay=0.99) # EMA для стабилизации весов

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Label smoothing
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_acc = 0.0

    print("\n⚡ Старт обучения эксперта нейтрофилов...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1:02d}/{EPOCHS}", leave=False)
        for step, (images, labels) in enumerate(pbar):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels) / GRAD_ACCUM_STEPS
            loss.backward()
            
            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                model_ema.update(model)
                
            running_loss += loss.item() * GRAD_ACCUM_STEPS
            
        scheduler.step()
        
        # ВАЛИДАЦИЯ (используем EMA модель)
        model_ema.module.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                # 4-Fold TTA (Test Time Augmentation)
                outputs = (
                    model_ema.module(images) + 
                    model_ema.module(torch.rot90(images, 1, [2, 3])) + 
                    model_ema.module(torch.rot90(images, 2, [2, 3])) + 
                    model_ema.module(torch.rot90(images, 3, [2, 3]))
                ) / 4.0
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model_ema.module.state_dict(), OUTPUT_WEIGHTS)
            print("   👉 Новый лучший чекпоинт сохранен.")

    print(f"\n🎉 Обучение завершено. Лучшая точность: {best_acc:.2f}%.")
    print(f"💾 Веса эксперта лежат тут: {OUTPUT_WEIGHTS}")

if __name__ == "__main__":
    train_expert()