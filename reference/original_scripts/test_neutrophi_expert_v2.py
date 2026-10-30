import os
import copy
import random
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import timm
import torchvision.transforms.functional as TF
from torchvision import transforms

from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================

DATA_DIR = "/vol/firstbatch/manual_cell_exports_1001"

CSV_PATH = os.path.join(
    DATA_DIR,
    "annotation_neutrophil_completed.csv"
)

MODEL_NAME = (
    "hf_hub:1aurent/vit_base_patch14_224.dinobloom"
)

# Старый эксперт используем как стартовую точку.
OLD_EXPERT = (
    "/vol/firstbatch/neutrophil_expert.pth"
)

# Новый fine-tuned expert.
OUTPUT_WEIGHTS = (
    "/vol/firstbatch/neutrophil_expert_finetuned.pth"
)

OUTPUT_DIR = (
    "/vol/firstbatch/neutrophil_expert_finetuning"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

PREDICTIONS_DIR = os.path.join(
    OUTPUT_DIR,
    "predictions",
)

os.makedirs(
    PREDICTIONS_DIR,
    exist_ok=True,
)


# ==============================================================================
# MODEL / TRAINING
# ==============================================================================

IMG_SIZE = 224

BATCH_SIZE = 8

GRAD_ACCUM_STEPS = 4

EPOCHS = 30

NUM_WORKERS = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42


# ==============================================================================
# FINE-TUNING PARAMETERS
# ==============================================================================

# Сколько последних transformer-блоков размораживаем.
UNFREEZE_LAST_BLOCKS = 4

# Learning rate для backbone.
BACKBONE_LR = 1e-5

# Learning rate для classification head.
HEAD_LR = 2e-4

WEIGHT_DECAY = 1e-2

LABEL_SMOOTHING = 0.05

# Насколько усиливаем редкий класс через sampler.
# 1.0 = обычный inverse-frequency sampler.
# 0.5 = более мягкий баланс.
SAMPLER_POWER = 0.5

# Сохраняем лучшую модель по этой метрике.
# Macro F1 лучше отражает проблему SiglecF+.
SELECTION_METRIC = "macro_f1"


# ==============================================================================
# CLASSES
# ==============================================================================

CLASSES = [
    "Immature_Neu",
    "Mature_Neu_SiglecF+",
    "Mature_Neu_SiglecF-",
]

NUM_CLASSES = len(CLASSES)

CLASS_TO_IDX = {
    cls_name: idx
    for idx, cls_name in enumerate(CLASSES)
}


# ==============================================================================
# REPRODUCIBILITY
# ==============================================================================

def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)


# ==============================================================================
# 1. DATASET
# ==============================================================================

def smart_pad(img):

    delta_w = IMG_SIZE - img.size[0]
    delta_h = IMG_SIZE - img.size[1]

    if delta_w <= 0 and delta_h <= 0:

        return img.resize(
            (IMG_SIZE, IMG_SIZE),
            Image.BILINEAR,
        )

    pad_w = max(
        0,
        delta_w // 2,
    )

    pad_h = max(
        0,
        delta_h // 2,
    )

    return TF.pad(
        img,
        (
            pad_w,
            pad_h,
            delta_w - pad_w,
            delta_h - pad_h,
        ),
        padding_mode="reflect",
    )


class NeutrophilDataset(Dataset):

    def __init__(
        self,
        df,
        base_dir,
        is_train=False,
    ):

        self.df = df.reset_index(
            drop=True
        )

        self.base_dir = base_dir
        self.is_train = is_train

        # ------------------------------------------------------------------
        # Training augmentation
        # ------------------------------------------------------------------

        if is_train:

            self.transform = transforms.Compose([

                transforms.RandomHorizontalFlip(
                    p=0.5
                ),

                transforms.RandomVerticalFlip(
                    p=0.5
                ),

                transforms.RandomRotation(
                    degrees=180,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    fill=0,
                ),

                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.15,
                            contrast=0.15,
                            saturation=0.20,
                            hue=0.04,
                        )
                    ],
                    p=0.7,
                ),

                transforms.RandomApply(
                    [
                        transforms.GaussianBlur(
                            kernel_size=3,
                            sigma=(0.1, 0.8),
                        )
                    ],
                    p=0.15,
                ),

                transforms.Lambda(
                    smart_pad
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],
                ),
            ])

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------

        else:

            self.transform = transforms.Compose([

                transforms.Lambda(
                    smart_pad
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],
                ),
            ])

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = os.path.join(
            self.base_dir,
            row["blind_filename"],
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        label = CLASS_TO_IDX[
            row["cell_type"]
        ]

        image = self.transform(
            image
        )

        return image, label


# ==============================================================================
# 2. MODEL
# ==============================================================================

def create_model():

    print(
        "\nCreating DinoBloom model..."
    )

    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=NUM_CLASSES,
        img_size=IMG_SIZE,
        global_pool="avg",
    )

    return model


# ==============================================================================
# 3. LOAD OLD EXPERT
# ==============================================================================

def load_old_expert(model):

    if not os.path.exists(
        OLD_EXPERT
    ):

        print(
            "\nOld expert not found."
        )

        print(
            "Starting from pretrained DinoBloom."
        )

        return model

    print(
        "\nLoading previous neutrophil expert:"
    )

    print(
        OLD_EXPERT
    )

    checkpoint = torch.load(
        OLD_EXPERT,
        map_location=DEVICE,
        weights_only=True,
    )

    result = model.load_state_dict(
        checkpoint,
        strict=False,
    )

    print(
        f"Missing keys: "
        f"{len(result.missing_keys)}"
    )

    print(
        f"Unexpected keys: "
        f"{len(result.unexpected_keys)}"
    )

    print(
        "✓ Previous expert loaded."
    )

    return model


# ==============================================================================
# 4. FREEZE / UNFREEZE STRATEGY
# ==============================================================================

def configure_finetuning(model):

    print(
        "\nConfiguring fine-tuning..."
    )

    # Сначала замораживаем всё.
    for param in model.parameters():

        param.requires_grad = False

    # ------------------------------------------------------------------
    # Последние transformer blocks
    # ------------------------------------------------------------------

    if hasattr(
        model,
        "blocks"
    ):

        total_blocks = len(
            model.blocks
        )

        first_block = max(
            0,
            total_blocks - UNFREEZE_LAST_BLOCKS,
        )

        print(
            f"Total ViT blocks: "
            f"{total_blocks}"
        )

        print(
            f"Unfreezing blocks "
            f"{first_block}–{total_blocks - 1}"
        )

        for block in model.blocks[
            first_block:
        ]:

            for param in block.parameters():

                param.requires_grad = True

    else:

        raise RuntimeError(
            "Model does not contain "
            "'blocks'. Cannot configure "
            "ViT fine-tuning."
        )

    # ------------------------------------------------------------------
    # Final normalization
    # ------------------------------------------------------------------

    if hasattr(
        model,
        "norm"
    ):

        for param in model.norm.parameters():

            param.requires_grad = True

    # ------------------------------------------------------------------
    # Classification head
    # ------------------------------------------------------------------

    for param in model.head.parameters():

        param.requires_grad = True

    # ------------------------------------------------------------------
    # Print trainable parameters
    # ------------------------------------------------------------------

    total_params = 0
    trainable_params = 0

    for param in model.parameters():

        num = param.numel()

        total_params += num

        if param.requires_grad:

            trainable_params += num

    print(
        f"\nTotal parameters: "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    print(
        f"Trainable fraction: "
        f"{100.0 * trainable_params / total_params:.2f}%"
    )


# ==============================================================================
# 5. CLASS BALANCING
# ==============================================================================

def create_weighted_sampler(train_df):

    labels = np.array([
        CLASS_TO_IDX[x]
        for x in train_df["cell_type"]
    ])

    class_counts = np.bincount(
        labels,
        minlength=NUM_CLASSES,
    )

    print(
        "\nTraining class distribution:"
    )

    for idx, cls in enumerate(CLASSES):

        print(
            f"  {cls:30s}: "
            f"{class_counts[idx]}"
        )

    # Мягкое inverse-frequency weighting.
    #
    # Не делаем полный 1/count,
    # потому что SiglecF+ и так легко
    # переусилить.
    class_weights = (
        1.0
        / np.power(
            class_counts + 1e-6,
            SAMPLER_POWER,
        )
    )

    sample_weights = class_weights[
        labels
    ]

    sample_weights = torch.as_tensor(
        sample_weights,
        dtype=torch.double,
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_df),
        replacement=True,
    )

    return sampler, class_counts


# ==============================================================================
# 6. LOSS
# ==============================================================================

class FocalLoss(nn.Module):

    def __init__(
        self,
        gamma=1.5,
        label_smoothing=0.05,
    ):

        super().__init__()

        self.gamma = gamma

        self.label_smoothing = (
            label_smoothing
        )

    def forward(
        self,
        logits,
        targets,
    ):

        ce = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        pt = torch.exp(
            -ce
        )

        focal = (
            (1.0 - pt)
            ** self.gamma
        ) * ce

        return focal.mean()


# ==============================================================================
# 7. TTA
# ==============================================================================

def predict_tta(
    model,
    images,
):

    outputs = []

    outputs.append(
        model(images)
    )

    outputs.append(
        model(
            torch.rot90(
                images,
                1,
                [2, 3],
            )
        )
    )

    outputs.append(
        model(
            torch.rot90(
                images,
                2,
                [2, 3],
            )
        )
    )

    outputs.append(
        model(
            torch.rot90(
                images,
                3,
                [2, 3],
            )
        )
    )

    return torch.stack(
        outputs,
        dim=0,
    ).mean(
        dim=0
    )


# ==============================================================================
# 8. VALIDATION
# ==============================================================================

def validate(
    model,
    loader,
    use_tta=False,
):

    model.eval()

    all_targets = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in tqdm(
            loader,
            desc=(
                "Validation TTA"
                if use_tta
                else "Validation"
            ),
            leave=False,
        ):

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            if use_tta:

                logits = predict_tta(
                    model,
                    images,
                )

            else:

                logits = model(
                    images
                )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predictions = torch.argmax(
                probabilities,
                dim=1,
            )

            all_targets.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

    y_true = np.asarray(
        all_targets
    )

    y_pred = np.asarray(
        all_predictions
    )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0,
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
    )

    confidence = probabilities.max(
        axis=1
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mean_confidence": float(
            confidence.mean()
        ),
        "median_confidence": float(
            np.median(confidence)
        ),
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": probabilities,
    }


# ==============================================================================
# 9. SAVE VALIDATION PREDICTIONS
# ==============================================================================

def save_predictions(
    results,
    val_df,
):

    df = val_df.reset_index(
        drop=True
    ).copy()

    df["True_Class"] = [
        CLASSES[x]
        for x in results["y_true"]
    ]

    df["Predicted_Class"] = [
        CLASSES[x]
        for x in results["y_pred"]
    ]

    df["Confidence"] = (
        results["probabilities"].max(
            axis=1
        )
    )

    for idx, cls in enumerate(CLASSES):

        safe_name = cls.replace(
            "+",
            "plus",
        ).replace(
            "-",
            "minus",
        )

        df[
            f"Prob_{safe_name}"
        ] = results[
            "probabilities"
        ][:, idx]

    path = os.path.join(
        PREDICTIONS_DIR,
        "validation_predictions.csv",
    )

    df.to_csv(
        path,
        index=False,
    )

    print(
        f"\nSaved predictions:"
        f"\n{path}"
    )

    return df


# ==============================================================================
# 10. MAIN TRAINING
# ==============================================================================

def train_expert():

    print(
        "=" * 90
    )

    print(
        "NEUTROPHIL EXPERT — FULL FINE-TUNING"
    )

    print(
        "=" * 90
    )

    print(
        f"Device:       {DEVICE}"
    )

    print(
        f"Dataset:      {CSV_PATH}"
    )

    print(
        f"Old expert:   {OLD_EXPERT}"
    )

    print(
        f"New expert:   {OUTPUT_WEIGHTS}"
    )

    print(
        f"Classes:      {CLASSES}"
    )

    print(
        f"Epochs:       {EPOCHS}"
    )

    print(
        f"Batch:        {BATCH_SIZE}"
    )

    print(
        f"Accumulation: {GRAD_ACCUM_STEPS}"
    )

    print(
        f"Unfreeze:     last {UNFREEZE_LAST_BLOCKS} blocks"
    )

    print(
        "=" * 90
    )


    # ==========================================================================
    # LOAD DATA
    # ==========================================================================

    print(
        "\nLoading annotation CSV..."
    )

    df = pd.read_csv(
        CSV_PATH
    )

    print(
        f"Total annotated cells: "
        f"{len(df)}"
    )

    print(
        "\nOriginal class distribution:"
    )

    print(
        df["cell_type"]
        .value_counts()
        .to_string()
    )


    # ==========================================================================
    # TRAIN / VALIDATION SPLIT
    # ==========================================================================

    print(
        "\nCreating validation split..."
    )

    train_df, val_df = train_test_split(
        df,
        test_size=0.15,
        stratify=df["cell_type"],
        random_state=SEED,
    )

    train_df = train_df.reset_index(
        drop=True
    )

    val_df = val_df.reset_index(
        drop=True
    )

    print(
        f"Training cells:   "
        f"{len(train_df)}"
    )

    print(
        f"Validation cells: "
        f"{len(val_df)}"
    )

    print(
        "\nValidation distribution:"
    )

    print(
        val_df["cell_type"]
        .value_counts()
        .to_string()
    )


    # ==========================================================================
    # DATASETS
    # ==========================================================================

    train_dataset = NeutrophilDataset(
        train_df,
        DATA_DIR,
        is_train=True,
    )

    val_dataset = NeutrophilDataset(
        val_df,
        DATA_DIR,
        is_train=False,
    )


    # ==========================================================================
    # SAMPLER
    # ==========================================================================

    train_sampler, class_counts = (
        create_weighted_sampler(
            train_df
        )
    )


    # ==========================================================================
    # DATALOADERS
    # ==========================================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )


    # ==========================================================================
    # MODEL
    # ==========================================================================

    model = create_model()

    # Загружаем старого эксперта.
    model = load_old_expert(
        model
    )

    model = model.to(
        DEVICE
    )

    configure_finetuning(
        model
    )


    # ==========================================================================
    # LOSS
    # ==========================================================================

    criterion = FocalLoss(
        gamma=1.5,
        label_smoothing=LABEL_SMOOTHING,
    )


    # ==========================================================================
    # OPTIMIZER
    # ==========================================================================

    backbone_parameters = []
    head_parameters = []

    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        if (
            name.startswith("head.")
            or name.startswith("fc_norm.")
        ):

            head_parameters.append(
                param
            )

        else:

            backbone_parameters.append(
                param
            )

    print(
        "\nOptimizer:"
    )

    print(
        f"Backbone parameters: "
        f"{sum(p.numel() for p in backbone_parameters):,}"
    )

    print(
        f"Head parameters: "
        f"{sum(p.numel() for p in head_parameters):,}"
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": BACKBONE_LR,
            },
            {
                "params": head_parameters,
                "lr": HEAD_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )


    # ==========================================================================
    # SCHEDULER
    # ==========================================================================

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS,
            eta_min=1e-7,
        )
    )


    # ==========================================================================
    # TRAINING
    # ==========================================================================

    best_metric = -np.inf

    best_epoch = 0

    history = []

    print(
        "\n"
        + "=" * 90
    )

    print(
        "STARTING FINE-TUNING"
    )

    print(
        "=" * 90
    )


    for epoch in range(
        EPOCHS
    ):

        model.train()

        running_loss = 0.0

        optimizer.zero_grad(
            set_to_none=True
        )


        pbar = tqdm(
            train_loader,
            desc=(
                f"Epoch "
                f"{epoch + 1:02d}/"
                f"{EPOCHS}"
            ),
            dynamic_ncols=True,
        )


        for step, (
            images,
            labels,
        ) in enumerate(
            pbar
        ):

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )


            logits = model(
                images
            )

            loss = criterion(
                logits,
                labels,
            )

            loss_for_backward = (
                loss
                / GRAD_ACCUM_STEPS
            )

            loss_for_backward.backward()


            if (
                (step + 1)
                % GRAD_ACCUM_STEPS
                == 0
                or
                (step + 1)
                == len(train_loader)
            ):

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )


            running_loss += (
                loss.item()
            )

            pbar.set_postfix(
                loss=f"{loss.item():.4f}"
            )


        scheduler.step()


        # ======================================================================
        # VALIDATION
        # ======================================================================

        val_results = validate(
            model,
            val_loader,
            use_tta=True,
        )


        # ======================================================================
        # REPORT
        # ======================================================================

        print(
            "\n"
            + "-" * 90
        )

        print(
            f"Epoch "
            f"{epoch + 1:02d}/{EPOCHS}"
        )

        print(
            f"Train loss:          "
            f"{running_loss / len(train_loader):.4f}"
        )

        print(
            f"Accuracy:             "
            f"{val_results['accuracy']:.4f}"
        )

        print(
            f"Balanced accuracy:    "
            f"{val_results['balanced_accuracy']:.4f}"
        )

        print(
            f"Macro F1:              "
            f"{val_results['macro_f1']:.4f}"
        )

        print(
            f"Weighted F1:           "
            f"{val_results['weighted_f1']:.4f}"
        )

        print(
            f"Mean confidence:       "
            f"{val_results['mean_confidence']:.4f}"
        )

        print(
            f"Median confidence:     "
            f"{val_results['median_confidence']:.4f}"
        )

        print(
            f"Backbone LR:           "
            f"{scheduler.get_last_lr()[0]:.2e}"
        )

        print(
            f"Head LR:               "
            f"{scheduler.get_last_lr()[1]:.2e}"
        )


        # ======================================================================
        # CLASSIFICATION REPORT
        # ======================================================================

        print(
            "\nClassification report:"
        )

        print(
            classification_report(
                val_results["y_true"],
                val_results["y_pred"],
                labels=list(
                    range(NUM_CLASSES)
                ),
                target_names=CLASSES,
                digits=3,
                zero_division=0,
            )
        )


        # ======================================================================
        # CONFUSION MATRIX
        # ======================================================================

        cm = confusion_matrix(
            val_results["y_true"],
            val_results["y_pred"],
            labels=list(
                range(NUM_CLASSES)
            ),
        )

        print(
            "Confusion matrix:"
        )

        print(
            pd.DataFrame(
                cm,
                index=CLASSES,
                columns=CLASSES,
            )
        )


        # ======================================================================
        # SAVE HISTORY
        # ======================================================================

        history_row = {

            "Epoch":
            epoch + 1,

            "Train_Loss":
            running_loss / len(
                train_loader
            ),

            "Accuracy":
            val_results["accuracy"],

            "Balanced_Accuracy":
            val_results[
                "balanced_accuracy"
            ],

            "Macro_F1":
            val_results[
                "macro_f1"
            ],

            "Weighted_F1":
            val_results[
                "weighted_f1"
            ],

            "Mean_Confidence":
            val_results[
                "mean_confidence"
            ],

            "Median_Confidence":
            val_results[
                "median_confidence"
            ],

            "Backbone_LR":
            scheduler.get_last_lr()[0],

            "Head_LR":
            scheduler.get_last_lr()[1],
        }

        history.append(
            history_row
        )

        history_df = pd.DataFrame(
            history
        )

        history_df.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "training_history.csv",
            ),
            index=False,
        )


        # ======================================================================
        # SAVE BEST MODEL
        # ======================================================================

        current_metric = (
            val_results["macro_f1"]
        )

        if (
            current_metric
            > best_metric
        ):

            best_metric = (
                current_metric
            )

            best_epoch = (
                epoch + 1
            )

            torch.save(
                model.state_dict(),
                OUTPUT_WEIGHTS,
            )

            print(
                "\n"
                "🔥 NEW BEST MODEL"
            )

            print(
                f"Macro F1: "
                f"{best_metric:.4f}"
            )

            print(
                f"Saved to:"
                f"\n{OUTPUT_WEIGHTS}"
            )


        print(
            "-" * 90
        )


    # ==========================================================================
    # FINAL
    # ==========================================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "FINE-TUNING FINISHED"
    )

    print(
        "=" * 90
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best Macro F1: "
        f"{best_metric:.4f}"
    )

    print(
        f"Final model:"
        f"\n{OUTPUT_WEIGHTS}"
    )

    print(
        f"Training history:"
        f"\n{os.path.join(OUTPUT_DIR, 'training_history.csv')}"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":

    train_expert()