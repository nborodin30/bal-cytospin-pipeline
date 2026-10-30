import os
import gc
import re
import time
import warnings
from pathlib import Path

import cv2
import czifile
import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms.functional as TF
from cellpose import models as cp_models
from PIL import Image
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from tqdm import tqdm
from torchvision import transforms

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================

# ------------------------------------------------------------------------------
# INPUT
# ------------------------------------------------------------------------------
SLIDES_DIR = Path(
    "/vol/firstbatch/First_batch_BAL_cytology"
)

# ------------------------------------------------------------------------------
# CLASSIFIER (PRODUCTION)
# ------------------------------------------------------------------------------
MODEL_CHECKPOINT = Path(
    "/vol/firstbatch/production_model/dinobloom_bal_production.pth"
)
MODEL_NAME = "hf_hub:1aurent/vit_base_patch14_224.dinobloom"

# ------------------------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------------------------
OUTPUT_DIR = Path(
    "/vol/firstbatch/full_slide_biological_analysis_final"
)

CELLS_DIR = OUTPUT_DIR / "cells"
SLIDE_STATS_DIR = OUTPUT_DIR / "slide_statistics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CELLS_DIR.mkdir(parents=True, exist_ok=True)
SLIDE_STATS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# CLASSES
# EXACT order used by the trained classifier
# ------------------------------------------------------------------------------
CLASSES = [
    "Macrophage",
    "Macrophage_with_particles",
    "Neutrophil",
    "Erythrocyte",
    "Lymphocyte",
    "Monocyte",
    "Eosinophil",
    "Junk",
]

# ------------------------------------------------------------------------------
# IMAGE / CELLPOSE
# ------------------------------------------------------------------------------
BLOCK_SIZE = 2500
SEG_PADDING = 256
MIN_CELL_AREA = 150

# ------------------------------------------------------------------------------
# DINO BLOOM INPUT
# ------------------------------------------------------------------------------
CROP_SIZE = 224
HALF = CROP_SIZE // 2
BATCH_SIZE = 32

# ------------------------------------------------------------------------------
# CELLPOSE PARAMETERS
# ------------------------------------------------------------------------------
CELLPOSE_FLOW_THRESHOLD = 0.9
CELLPOSE_CELLPROB_THRESHOLD = 0.0

# ------------------------------------------------------------------------------
# NANOTUBE / DARK PARTICLE DETECTION
# ------------------------------------------------------------------------------
RGB_DARK_THRESH = 75
MIN_PEAK_DIST = 2
MIN_CNT_FOR_PARTICLE_CLASS = 3

# НОВОЕ ПРАВИЛО: Если площадь черного куска больше этого значения (в пикселях), 
# мы считаем весь этот гигантский кусок за ОДНУ трубку/агломерат.
# Картинка у нас 224x224 (50176 px). Клетка занимает примерно 4000-8000 px.
LARGE_PARTICLE_AREA_THRESH = 2000 

# ------------------------------------------------------------------------------
# BIOLOGICAL CELL FILTER
# ------------------------------------------------------------------------------
MIN_SATURATION = 18
MIN_CONTRAST = 20

# ------------------------------------------------------------------------------
# DUPLICATE REMOVAL
# ------------------------------------------------------------------------------
NMS_IOU_THRESHOLD = 0.50

# ------------------------------------------------------------------------------
# SLIDES EXCLUDED FROM ANALYSIS
# ------------------------------------------------------------------------------
EXCLUDED_SLIDES = {
    "LHI_Kutschke_Slide_989-1a",
    "LHI_Kutschke_BAL_cells_991-1b_14",
    "LHI_Kutschke_BAL_cells_991-2a_15",
}

# ------------------------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------------------------
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==============================================================================
# 1. DINO BLOOM PREPROCESSING
# ==============================================================================

def smart_pad_transform(img):
    delta_w = CROP_SIZE - img.size[0]
    delta_h = CROP_SIZE - img.size[1]

    if delta_w <= 0 and delta_h <= 0:
        return img.resize(
            (CROP_SIZE, CROP_SIZE),
            Image.BILINEAR,
        )

    pad_w = max(0, delta_w // 2)
    pad_h = max(0, delta_h // 2)

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

PREPROCESS = transforms.Compose([
    transforms.Lambda(smart_pad_transform),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ==============================================================================
# 2. MODEL
# ==============================================================================

def create_classifier():
    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=len(CLASSES),
        img_size=CROP_SIZE,
        global_pool="avg",
    )
    return model

def load_classifier():
    print("\nLoading DinoBloom-Base Production classifier...")

    if not MODEL_CHECKPOINT.exists():
        raise FileNotFoundError(
            "Classifier checkpoint not found:\n"
            f"{MODEL_CHECKPOINT}"
        )

    model = create_classifier()

    checkpoint = torch.load(
        MODEL_CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint,
        strict=False,
    )

    model.to(DEVICE)
    model.eval()

    print("✓ DinoBloom-Base Production loaded")

    return model


# ==============================================================================
# 3. CELLPOSE
# ==============================================================================

print("=" * 80)
print("FULL-SLIDE BIOLOGICAL ANALYSIS (PRODUCTION)")
print("=" * 80)
print(f"Device: {DEVICE}")
print(f"Slides directory: {SLIDES_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Classifier: DinoBloom-Base Production")
print(f"Checkpoint: {MODEL_CHECKPOINT}")
print("=" * 80)

print("\nInitializing Cellpose-SAM...")

SEG_MODEL = cp_models.CellposeModel(
    gpu=torch.cuda.is_available(),
    device=DEVICE,
    model_type="cpsam",
)

print("✓ Cellpose-SAM ready.")


# ==============================================================================
# 4. IMAGE READING
# ==============================================================================

def load_czi_image(filepath):
    raw = czifile.imread(
        str(filepath)
    ).squeeze()

    img = raw

    while img.ndim > 3:
        img = img[0]

    if img.ndim == 3:
        if img.shape[0] in [1, 2, 3, 4]:
            img = np.transpose(
                img,
                (1, 2, 0),
            )
        elif img.shape[-1] not in [1, 2, 3, 4]:
            raise RuntimeError(
                f"Unexpected image shape: {img.shape}"
            )

    elif img.ndim == 2:
        img = np.stack(
            [img, img, img],
            axis=-1,
        )

    else:
        raise RuntimeError(
            f"Unexpected CZI dimensions: {img.shape}"
        )

    if img.shape[-1] == 1:
        img = np.repeat(
            img,
            3,
            axis=-1,
        )

    if img.shape[-1] > 3:
        img = img[:, :, :3]

    if img.dtype != np.uint8:
        img_max = img.max()

        if img_max > 0:
            img = (
                img.astype(np.float32)
                / float(img_max)
                * 255.0
            ).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    return img


# ==============================================================================
# 5. BIOLOGICAL CELL FILTER
# ==============================================================================

def is_valid_biological_cell(
    crop_rgb,
    min_saturation=MIN_SATURATION,
    min_contrast=MIN_CONTRAST,
):
    hsv = cv2.cvtColor(
        crop_rgb,
        cv2.COLOR_RGB2HSV,
    )

    saturation = hsv[:, :, 1]

    if (
        np.mean(saturation) < min_saturation
        and np.max(saturation) < 40
    ):
        return False

    gray = cv2.cvtColor(
        crop_rgb,
        cv2.COLOR_RGB2GRAY,
    )

    contrast = (
        np.percentile(gray, 95)
        - np.percentile(gray, 5)
    )

    if contrast < min_contrast:
        return False

    return True


# ==============================================================================
# 6. NANOTUBE / DARK PARTICLE DETECTION
# ==============================================================================

def detect_nanotubes(crop_rgb, cell_mask=None):
    r, g, b = crop_rgb[:, :, 0], crop_rgb[:, :, 1], crop_rgb[:, :, 2]
    carbon_mask = (r < RGB_DARK_THRESH) & (g < RGB_DARK_THRESH) & (b < RGB_DARK_THRESH)

    if cell_mask is not None:
        carbon_mask &= (cell_mask > 0)

    if not np.any(carbon_mask):
        return 0, []

    # 1. Находим все черные пятна (контуры)
    mask_uint8 = carbon_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    inv_gray = 255 - blurred

    total_particles = 0
    all_peaks = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Создаем маску конкретно для этого пятна
        blob_mask = np.zeros_like(carbon_mask, dtype=np.uint8)
        cv2.drawContours(blob_mask, [cnt], -1, 1, thickness=cv2.FILLED)

        # 2. Проверяем размер
        if area >= LARGE_PARTICLE_AREA_THRESH:
            # Огромное черное пятно -> считаем его за ОДНУ нанотрубку/агломерат
            total_particles += 1
            
            # Сохраняем центр массы этого куска
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cy = int(M["m01"] / M["m00"])
                cx = int(M["m10"] / M["m00"])
                all_peaks.append([cy, cx])
        else:
            # Маленькое пятно -> ищем локальные пики внутри него (старый метод)
            peaks = peak_local_max(
                inv_gray,
                min_distance=MIN_PEAK_DIST,
                labels=blob_mask.astype(bool)
            )
            total_particles += len(peaks)
            if len(peaks) > 0:
                all_peaks.extend(peaks)

    return total_particles, np.array(all_peaks) if len(all_peaks) > 0 else []


# ==============================================================================
# 7. SLIDE METADATA
# ==============================================================================

def infer_experimental_metadata(filepath):
    filepath = Path(filepath)
    filename = filepath.stem

    treatment_mapping = {
        "151-1": "HCC", "151-2": "HCC", "151-3": "HCC",
        "989-1": "Sham 12h", "989-2": "Sham 12h", "989-3": "Sham 12h",
        "991-1": "Sham 6d", "991-2": "Sham 6d", "991-3": "Sham 6d",
        "993-1": "Sham 28d", "993-2": "Sham 28d", "993-3": "Sham 28d",
        "990-1": "Ptx90 12h", "990-2": "Ptx90 12h", "990-3": "Ptx90 12h",
        "992-1": "Ptx90 6d", "992-2": "Ptx90 6d", "992-3": "Ptx90 6d",
        "994-1": "Ptx90 28d", "994-2": "Ptx90 28d", "994-3": "Ptx90 28d",
        "995-1": "Mitsui-7 12h", "995-2": "Mitsui-7 12h", "995-3": "Mitsui-7 12h",
        "997-1": "Mitsui-7 6d", "997-2": "Mitsui-7 6d", "997-3": "Mitsui-7 6d",
        "999-1": "Mitsui-7 28d", "999-2": "Mitsui-7 28d", "999-3": "Mitsui-7 28d",
        "996-1": "DWCNT 12h", "996-2": "DWCNT 12h", "996-3": "DWCNT 12h",
        "998-1": "DWCNT 6d", "998-2": "DWCNT 6d", "998-3": "DWCNT 6d",
        "1000-1": "DWCNT 28d", "1000-2": "DWCNT 28d",
        "1001-1": "LPS 12h", "1001-2": "LPS 12h", "1001-3": "LPS 12h",
        "1002-1": "LPS 5d", "1002-2": "LPS 5d", "1002-3": "LPS 5d",
        "1003-1": "LPS 28d", "1003-2": "LPS 28d",
    }

    match = re.search(r"(K\d{3,4}-\d+)[a-zA-Z]?(?:_|$)", filename, re.IGNORECASE)
    if match:
        slide_id = match.group(1).upper()
        mapping_key = slide_id[1:]
    else:
        match = re.search(r"(\d{3,4}-\d+)[a-zA-Z]?(?:_|$)", filename)
        if match:
            mapping_key = match.group(1)
            slide_id = mapping_key.upper()
        else:
            raise RuntimeError(f"Could not determine MB6 ID from CZI filename:\n{filepath}")

    if mapping_key not in treatment_mapping:
        raise RuntimeError(f"MB6 ID '{slide_id}' extracted but not found in mapping.\nFile: {filepath}")

    treatment_timepoint = treatment_mapping[mapping_key]

    if treatment_timepoint == "HCC":
        treatment = "HCC"
        timepoint = "NA"
    else:
        match = re.match(r"(.+?)\s+(\d+(?:\.\d+)?[hHdD])$", treatment_timepoint)
        if not match:
            raise RuntimeError(f"Invalid treatment mapping for '{slide_id}': '{treatment_timepoint}'")
        treatment = match.group(1)
        timepoint = match.group(2)

    experimental_group = f"{treatment}_{timepoint}" if timepoint != "NA" else treatment

    return {
        "MB6_ID": f"MB6 {slide_id}",
        "Treatment": treatment,
        "Timepoint": timepoint,
        "Experimental_Group": experimental_group,
    }


# ==============================================================================
# 8. TILE GENERATION
# ==============================================================================

def generate_tiles(H, W):
    for y_start in range(0, H, BLOCK_SIZE):
        for x_start in range(0, W, BLOCK_SIZE):
            y_end = min(H, y_start + BLOCK_SIZE)
            x_end = min(W, x_start + BLOCK_SIZE)
            yield (y_start, y_end, x_start, x_end)


# ==============================================================================
# 9. CELLPOSE TILE SEGMENTATION
# ==============================================================================

def segment_tile(img, y_start, y_end, x_start, x_end):
    chunk = img[y_start:y_end, x_start:x_end]
    if chunk.size == 0:
        return []

    mean_intensity = float(chunk.mean())
    if mean_intensity > 245 or mean_intensity < 10:
        return []

    padded = cv2.copyMakeBorder(
        chunk, SEG_PADDING, SEG_PADDING, SEG_PADDING, SEG_PADDING, cv2.BORDER_REFLECT
    )

    result = SEG_MODEL.eval(
        padded, diameter=None, channels=[0, 0],
        cellprob_threshold=CELLPOSE_CELLPROB_THRESHOLD,
        flow_threshold=CELLPOSE_FLOW_THRESHOLD,
    )

    masks = result[0] if isinstance(result, tuple) else result
    masks = masks[
        SEG_PADDING:SEG_PADDING + chunk.shape[0],
        SEG_PADDING:SEG_PADDING + chunk.shape[1],
    ]

    props = regionprops(masks)
    cells = []

    for prop in props:
        if prop.area < MIN_CELL_AREA:
            continue
        cy, cx = prop.centroid
        cy = int(round(cy))
        cx = int(round(cx))
        minr, minc, maxr, maxc = prop.bbox

        cells.append({
            "local_cy": cy, "local_cx": cx,
            "minr": int(minr), "minc": int(minc),
            "maxr": int(maxr), "maxc": int(maxc),
            "cell_label": int(prop.label),
            "area": int(prop.area),
        })

    del padded, masks, props, result
    return cells


# ==============================================================================
# 10. GLOBAL NMS
# ==============================================================================

def bbox_iou(box_a, box_b):
    a_minr, a_minc, a_maxr, a_maxc = box_a
    b_minr, b_minc, b_maxr, b_maxc = box_b

    inter_minr = max(a_minr, b_minr)
    inter_minc = max(a_minc, b_minc)
    inter_maxr = min(a_maxr, b_maxr)
    inter_maxc = min(a_maxc, b_maxc)

    if inter_maxr <= inter_minr or inter_maxc <= inter_minc:
        return 0.0

    intersection = (inter_maxr - inter_minr) * (inter_maxc - inter_minc)
    area_a = (a_maxr - a_minr) * (a_maxc - a_minc)
    area_b = (b_maxr - b_minr) * (b_maxc - b_minc)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0

def deduplicate_cells(cells):
    """
    Remove duplicate cell detections using spatial tile-based comparison.

    Only cells from the same tile or one of the 8 neighboring tiles
    are compared. Larger detections are processed first and retained
    when overlapping detections are found.
    """

    if not cells:
        return []

    print(
        f"Starting spatial deduplication of {len(cells):,} detections...",
        flush=True
    )

    # Process larger cells first
    cells = sorted(cells, key=lambda x: x["area"], reverse=True)

    # Spatial index:
    # (tile_y, tile_x) -> list of already accepted cells
    spatial_index = {}

    kept = []

    for i, candidate in enumerate(cells):

        tile_y = candidate["tile_y"]
        tile_x = candidate["tile_x"]

        # Collect cells from the current tile and its 8 neighbors
        neighboring_cells = []

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):

                key = (tile_y + dy, tile_x + dx)

                if key in spatial_index:
                    neighboring_cells.extend(spatial_index[key])

        candidate_bbox = (
            candidate["minr"],
            candidate["minc"],
            candidate["maxr"],
            candidate["maxc"],
        )

        is_duplicate = False

        for existing in neighboring_cells:

            existing_bbox = (
                existing["minr"],
                existing["minc"],
                existing["maxr"],
                existing["maxc"],
            )

            if bbox_iou(candidate_bbox, existing_bbox) > 0.5:
                is_duplicate = True
                break

        if not is_duplicate:

            kept.append(candidate)

            key = (tile_y, tile_x)

            if key not in spatial_index:
                spatial_index[key] = []

            spatial_index[key].append(candidate)

        # Progress reporting
        if (i + 1) % 5000 == 0:
            print(
                f"  Deduplication: "
                f"{i + 1:,}/{len(cells):,} processed, "
                f"{len(kept):,} kept",
                flush=True
            )

    print(
        f"Deduplication complete: "
        f"{len(cells):,} → {len(kept):,} cells",
        flush=True
    )

    return kept

# ==============================================================================
# 11. CELL CROP EXTRACTION
# ==============================================================================

def extract_cell_crop(img, cell):
    H, W = img.shape[:2]
    cy = cell["global_cy"]
    cx = cell["global_cx"]

    y1, y2 = cy - HALF, cy + HALF
    x1, x2 = cx - HALF, cx + HALF

    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - H)
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - W)

    crop = img[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]

    if pad_top or pad_bottom or pad_left or pad_right:
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT
        )

    return crop


# ==============================================================================
# 12. CLASSIFICATION
# ==============================================================================

def predict_with_model(model, crops):
    if not crops:
        return np.empty((0, len(CLASSES)), dtype=np.float32)

    probabilities = []

    for start in tqdm(range(0, len(crops), BATCH_SIZE), desc="DinoBloom", unit="batch", dynamic_ncols=True):
        batch_crops = crops[start:start + BATCH_SIZE]
        tensors = [PREPROCESS(Image.fromarray(crop)) for crop in batch_crops]
        batch = torch.stack(tensors).to(DEVICE, non_blocking=True)

        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)

        probabilities.append(probs.detach().cpu().numpy().astype(np.float32))

        del batch, tensors, logits, probs

    return np.concatenate(probabilities, axis=0)


# ==============================================================================
# 13. SLIDE STATISTICS
# ==============================================================================

def calculate_slide_statistics(df, slide_name, metadata):
    total_cells = len(df)

    result = {
        "Slide": slide_name,
        "Treatment": metadata["Treatment"],
        "Timepoint": metadata["Timepoint"],
        "Experimental_Group": metadata["Experimental_Group"],
        "Classifier": "DinoBloom-Base_Production",
        "Total_Detected_Cells": total_cells,
    }

    counts = df["Class"].value_counts()
    for cls in CLASSES:
        result[f"Count_{cls}"] = int(counts.get(cls, 0))

    for cls in CLASSES:
        count = counts.get(cls, 0)
        result[f"Share_{cls}_AllCells_pct"] = (100.0 * count / total_cells if total_cells > 0 else 0.0)

    wbc_classes = [
        "Macrophage", "Macrophage_with_particles", "Neutrophil",
        "Lymphocyte", "Monocyte", "Eosinophil",
    ]

    wbc_mask = ~df["Class"].isin(["Junk", "Erythrocyte"])
    df_wbc = df[wbc_mask]
    total_wbc = len(df_wbc)

    result["Total_WBC"] = total_wbc
    result["WBC_Share_of_AllCells_pct"] = (100.0 * total_wbc / total_cells if total_cells > 0 else 0.0)

    wbc_counts = df_wbc["Class"].value_counts()
    for cls in wbc_classes:
        count = wbc_counts.get(cls, 0)
        result[f"BAL_Differential_{cls}_pct"] = (100.0 * count / total_wbc if total_wbc > 0 else 0.0)

    macrophage_mask = df["Class"].isin(["Macrophage", "Macrophage_with_particles"])
    macrophages = df[macrophage_mask]
    macrophage_count = len(macrophages)

    particle_positive = df[df["Class"] == "Macrophage_with_particles"]
    particle_positive_count = len(particle_positive)

    result["Total_Macrophages"] = macrophage_count
    result["Macrophages_with_Particles"] = particle_positive_count
    result["Particle_Positive_Macrophage_Share_pct"] = (
        100.0 * particle_positive_count / macrophage_count if macrophage_count > 0 else 0.0
    )

    if "Nanotubes_Count" in df.columns:
        result["Total_Detected_Particles"] = int(df["Nanotubes_Count"].sum())
        result["Mean_Particles_AllCells"] = float(df["Nanotubes_Count"].mean())
        result["Median_Particles_AllCells"] = float(df["Nanotubes_Count"].median())

        if macrophage_count > 0:
            result["Mean_Particles_per_Macrophage"] = float(macrophages["Nanotubes_Count"].mean())
            result["Median_Particles_per_Macrophage"] = float(macrophages["Nanotubes_Count"].median())
        else:
            result["Mean_Particles_per_Macrophage"] = 0.0
            result["Median_Particles_per_Macrophage"] = 0.0

    if "Model_Confidence" in df.columns and total_cells > 0:
        result["Mean_Model_Confidence"] = float(df["Model_Confidence"].mean())
        result["Median_Model_Confidence"] = float(df["Model_Confidence"].median())

    return pd.DataFrame([result])


# ==============================================================================
# 14. PROCESS ONE SLIDE
# ==============================================================================

def process_slide(filepath, classifier):
    filepath = Path(filepath)
    slide_name = filepath.stem.replace(" ", "_")

    print("\n" + "=" * 80)
    print(f"PROCESSING: {slide_name}")
    print("=" * 80)

    start_time = time.time()
    metadata = infer_experimental_metadata(filepath)

    print(f"Treatment: {metadata['Treatment']}")
    print(f"Timepoint: {metadata['Timepoint']}")

    print("\nLoading CZI...")
    img = load_czi_image(filepath)
    H, W = img.shape[:2]
    print(f"Image size: {W:,} × {H:,} px")

    print("\nSegmenting complete slide...")
    raw_cells = []
    tiles = list(generate_tiles(H, W))
    total_tiles = len(tiles)

    for tile_index, (y_start, y_end, x_start, x_end) in enumerate(
        tqdm(tiles, total=total_tiles, desc=f"Cellpose {slide_name}", unit="tile", dynamic_ncols=True),
        start=1,
    ):
        tile_cells = segment_tile(img, y_start, y_end, x_start, x_end)

        for cell in tile_cells:
            cell["global_cy"] = y_start + cell["local_cy"]
            cell["global_cx"] = x_start + cell["local_cx"]
            cell["global_minr"] = y_start + cell["minr"]
            cell["global_minc"] = x_start + cell["minc"]
            cell["global_maxr"] = y_start + cell["maxr"]
            cell["global_maxc"] = x_start + cell["maxc"]
            cell["tile_y"] = y_start
            cell["tile_x"] = x_start
            raw_cells.append(cell)

        if tile_index == 1 or tile_index % 5 == 0 or tile_index == total_tiles:
            percent = 100.0 * tile_index / total_tiles
            print(f"[Segmentation] {tile_index:,}/{total_tiles:,} ({percent:.1f}%) | Raw detections: {len(raw_cells):,}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nRaw Cellpose detections: {len(raw_cells):,}")

    print("\nRemoving duplicate detections...")
    cells = deduplicate_cells(raw_cells)
    print(f"Unique segmented cells: {len(cells):,}")

    del raw_cells
    gc.collect()

    print("\nExtracting cell crops...")
    crops = []
    valid_cells = []

    for cell in tqdm(cells, desc=f"Crops {slide_name}", unit="cell", dynamic_ncols=True):
        crop = extract_cell_crop(img, cell)
        if not is_valid_biological_cell(crop):
            continue
        crops.append(crop)
        valid_cells.append(cell)

    cells = valid_cells
    print(f"Cells after biological filtering: {len(cells):,}")

    print("\n" + "-" * 80)
    print(f"Running DinoBloom-Base Production")
    print(f"Classifying {len(crops):,} cells...")
    print("-" * 80)

    probabilities = predict_with_model(classifier, crops)
    print("\nClassification finished.")
    print(f"Probability matrix: {probabilities.shape}")

    records = []
    for i, cell in enumerate(cells):
        probs = probabilities[i]
        predicted_idx = int(np.argmax(probs))
        initial_class = CLASSES[predicted_idx]
        model_confidence = float(probs[predicted_idx])
        particle_count = 0

        if initial_class in ["Macrophage", "Macrophage_with_particles", "Monocyte"]:
            particle_count, _ = detect_nanotubes(crops[i])

        final_class = initial_class
        if particle_count >= MIN_CNT_FOR_PARTICLE_CLASS:
            final_class = "Macrophage_with_particles"

        record = {
            "Slide": slide_name,
            "Cell_ID": i + 1,
            "Class": final_class,
            "Model_Predicted_Class": initial_class,
            "Model_Confidence": model_confidence,
            "Nanotubes_Count": particle_count,
            "Centroid_Y": cell["global_cy"],
            "Centroid_X": cell["global_cx"],
            "BBox_MinY": cell["global_minr"],
            "BBox_MinX": cell["global_minc"],
            "BBox_MaxY": cell["global_maxr"],
            "BBox_MaxX": cell["global_maxc"],
            "Segmented_Area": cell["area"],
        }

        for class_index, class_name in enumerate(CLASSES):
            record[f"Prob_{class_name}"] = float(probs[class_index])

        records.append(record)

    df_cells = pd.DataFrame(records)
    for key, value in metadata.items():
        df_cells[key] = value

    first_columns = [
        "Slide", "MB6_ID", "Treatment", "Timepoint", "Experimental_Group",
        "Cell_ID", "Class", "Model_Predicted_Class", "Model_Confidence",
        "Nanotubes_Count", "Centroid_Y", "Centroid_X", "BBox_MinY", "BBox_MinX",
        "BBox_MaxY", "BBox_MaxX", "Segmented_Area",
    ]

    remaining_columns = [c for c in df_cells.columns if c not in first_columns]
    df_cells = df_cells[first_columns + remaining_columns]

    cells_path = CELLS_DIR / f"{slide_name}_cells.csv"
    df_cells.to_csv(cells_path, index=False)
    print(f"\nSaved cell table:\n{cells_path}")

    slide_stats = calculate_slide_statistics(df_cells, slide_name, metadata)
    slide_stats_path = SLIDE_STATS_DIR / f"{slide_name}_summary.csv"
    slide_stats.to_csv(slide_stats_path, index=False)
    print(f"Saved slide summary:\n{slide_stats_path}")

    print("\nClass distribution:")
    print(df_cells["Class"].value_counts().to_string())
    print("\nModel-predicted class distribution:")
    print(df_cells["Model_Predicted_Class"].value_counts().to_string())
    print(f"\nMean model confidence: {df_cells['Model_Confidence'].mean():.4f}")
    print(f"Median model confidence: {df_cells['Model_Confidence'].median():.4f}")
    print(f"Mean nanotubes/cell: {df_cells['Nanotubes_Count'].mean():.4f}")
    print(f"Median nanotubes/cell: {df_cells['Nanotubes_Count'].median():.4f}")

    del img, crops, cells, probabilities, df_cells
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print(f"\nFinished {slide_name} in {elapsed / 60:.1f} min")

    return slide_stats


# ==============================================================================
# 15. FIND ALL CZI SLIDES
# ==============================================================================

def find_all_czi_slides():
    slides = sorted(
        p for p in SLIDES_DIR.rglob("*.czi")
        if p.stem.replace(" ", "_") not in EXCLUDED_SLIDES
    )
    print(f"\nFound {len(slides)} CZI files for analysis.")
    print(f"Excluded {len(EXCLUDED_SLIDES)} non-analysis CZI files.")
    return slides


# ==============================================================================
# 16. GROUP-LEVEL STATISTICS
# ==============================================================================

def create_group_statistics(slide_df):
    if slide_df.empty:
        return

    excluded_columns = {
        "Slide", "Treatment", "Timepoint", "Experimental_Group",
        "Classifier", "Validation_Fold", "Model_Fold",
    }

    numeric_columns = [
        c for c in slide_df.columns
        if c not in excluded_columns and pd.api.types.is_numeric_dtype(slide_df[c])
    ]

    grouped = (
        slide_df.groupby(["Treatment", "Timepoint", "Experimental_Group"], dropna=False)[numeric_columns]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
    )

    grouped.columns = [
        (f"{col[0]}_{col[1]}" if isinstance(col, tuple) and col[1] else str(col[0]) if isinstance(col, tuple) else str(col))
        for col in grouped.columns
    ]

    path = OUTPUT_DIR / "GROUP_LEVEL_STATISTICS.csv"
    grouped.to_csv(path, index=False)
    print(f"\nSaved group-level statistics:\n{path}")


# ==============================================================================
# 17. TREATMENT x TIME STATISTICS
# ==============================================================================

def create_treatment_time_statistics(slide_df):
    if slide_df.empty:
        return
    if not {"Treatment", "Timepoint"}.issubset(slide_df.columns):
        return

    excluded_columns = {
        "Slide", "Treatment", "Timepoint", "Experimental_Group",
        "Classifier", "Validation_Fold", "Model_Fold",
    }

    numeric_columns = [
        c for c in slide_df.columns
        if c not in excluded_columns and pd.api.types.is_numeric_dtype(slide_df[c])
    ]

    grouped = (
        slide_df.groupby(["Treatment", "Timepoint"], dropna=False)[numeric_columns]
        .agg(["mean", "std", "median"])
        .reset_index()
    )

    grouped.columns = [
        (f"{col[0]}_{col[1]}" if isinstance(col, tuple) and col[1] else str(col[0]) if isinstance(col, tuple) else str(col))
        for col in grouped.columns
    ]

    path = OUTPUT_DIR / "TREATMENT_TIME_STATISTICS.csv"
    grouped.to_csv(path, index=False)
    print(f"\nSaved treatment × time statistics:\n{path}")


# ==============================================================================
# 18. PROCESS ALL SLIDES
# ==============================================================================

def process_all_slides():
    slides = find_all_czi_slides()

    if not slides:
        raise RuntimeError(
            f"No CZI slides found in:\n{SLIDES_DIR}"
        )

    print("\nValidating slide metadata...")

    metadata_errors = []

    for filepath in slides:
        try:
            infer_experimental_metadata(filepath)

        except Exception as e:
            metadata_errors.append({
                "Slide": filepath.stem,
                "Path": str(filepath),
                "Error": repr(e),
            })

    if metadata_errors:
        print("\nMetadata validation failed:")

        for item in metadata_errors:
            print(
                f"  {item['Slide']}: "
                f"{item['Error']}"
            )

        raise RuntimeError(
            "Refusing to start because some slides could not be "
            "mapped to treatment/timepoint."
        )

    print("✓ All slide metadata successfully mapped.")

    classifier = load_classifier()

    failed_slides = []
    processed_this_run = 0
    skipped_slides = 0

    overall_start = time.time()

    for index, filepath in enumerate(slides, start=1):

        # process_slide() saves files with underscores,
        # while some CZI filenames contain spaces.
        slide_name = filepath.stem.replace(" ", "_")

        cells_output = (
            OUTPUT_DIR
            / "cells"
            / f"{slide_name}_cells.csv"
        )

        summary_output = (
            OUTPUT_DIR
            / "slide_statistics"
            / f"{slide_name}_summary.csv"
        )

        # ----------------------------------------------------------
        # Resume logic
        # ----------------------------------------------------------

        print(f"\nCHECKING RESUME: {slide_name}")
        print(
            f"Cells exists: "
            f"{cells_output.exists()}"
        )
        print(
            f"Summary exists: "
            f"{summary_output.exists()}"
        )

        if (
            cells_output.exists()
            and summary_output.exists()
        ):

            print("\n" + "-" * 80)
            print(
                f"SKIPPING ALREADY PROCESSED SLIDE "
                f"{index}/{len(slides)}"
            )
            print(slide_name)
            print("-" * 80)

            skipped_slides += 1
            continue

        # ----------------------------------------------------------
        # Process slide
        # ----------------------------------------------------------

        print("\n\n" + "#" * 80)
        print(f"SLIDE {index}/{len(slides)}")
        print(filepath)
        print(
            "GLOBAL CLASSIFIER: "
            "DinoBloom-Base Production"
        )
        print("#" * 80)

        try:
            process_slide(
                filepath,
                classifier
            )

            processed_this_run += 1

        except Exception as e:

            print("\n❌ FAILED:")
            print(filepath)
            print(repr(e))

            failed_slides.append({
                "Slide": slide_name,
                "Path": str(filepath),
                "Error": repr(e),
            })

            continue

    # --------------------------------------------------------------
    # Free classifier memory
    # --------------------------------------------------------------

    del classifier
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------------
    # Rebuild combined statistics from ALL slide summaries
    # --------------------------------------------------------------

    summary_files = sorted(
        (
            OUTPUT_DIR
            / "slide_statistics"
        ).glob("*_summary.csv")
    )

    if summary_files:

        print(
            f"\nFound {len(summary_files)} "
            "processed slide summaries."
        )

        combined_slides = pd.concat(
            [
                pd.read_csv(summary_file)
                for summary_file in summary_files
            ],
            ignore_index=True,
        )

        combined_path = (
            OUTPUT_DIR
            / "ALL_SLIDE_STATISTICS.csv"
        )

        combined_slides.to_csv(
            combined_path,
            index=False,
        )

        print(
            f"\nSaved combined slide statistics:\n"
            f"{combined_path}"
        )

        create_group_statistics(
            combined_slides
        )

        create_treatment_time_statistics(
            combined_slides
        )

    else:
        print(
            "\nNo slide summary files were found."
        )

    # --------------------------------------------------------------
    # Save failed slides
    # --------------------------------------------------------------

    if failed_slides:

        failed_df = pd.DataFrame(
            failed_slides
        )

        failed_path = (
            OUTPUT_DIR
            / "FAILED_SLIDES.csv"
        )

        failed_df.to_csv(
            failed_path,
            index=False,
        )

        print(
            f"\nFailed slides saved to:\n"
            f"{failed_path}"
        )

    # --------------------------------------------------------------
    # Final report
    # --------------------------------------------------------------

    elapsed = time.time() - overall_start

    print("\n" + "=" * 80)
    print("FULL DATASET ANALYSIS FINISHED")
    print("=" * 80)

    print(
        f"Total runtime: "
        f"{elapsed / 3600:.2f} h"
    )

    print(
        f"Processed in this run: "
        f"{processed_this_run}"
    )

    print(
        f"Skipped already processed: "
        f"{skipped_slides}"
    )

    print(
        f"Total completed slides: "
        f"{len(summary_files)}"
    )

    print(
        f"Failed in this run: "
        f"{len(failed_slides)}"
    )

    print(
        f"Output directory:\n"
        f"{OUTPUT_DIR}"
    )

    print("=" * 80)


if __name__ == "__main__":
    process_all_slides()
