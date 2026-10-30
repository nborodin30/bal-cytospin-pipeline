#!/usr/bin/env python3

import os
import gc
import re
import time
import warnings
from pathlib import Path

import czifile
import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms.functional as TF

from PIL import Image
from tqdm import tqdm
from torchvision import transforms


warnings.filterwarnings("ignore")


# =============================================================================
# 0. CONFIGURATION
# =============================================================================

# -----------------------------------------------------------------------------
# ORIGINAL CZI SLIDES
# -----------------------------------------------------------------------------

SLIDES_DIR = Path(
    "/vol/firstbatch/First_batch_BAL_cytology"
)


# -----------------------------------------------------------------------------
# EXISTING FULL-SLIDE ANALYSIS
#
# Contains:
#   ALL_SLIDE_STATISTICS.csv
#   cells/
#       <slide>_cells.csv
# -----------------------------------------------------------------------------

BIOLOGICAL_ANALYSIS_DIR = Path(
    "/vol/firstbatch/full_slide_biological_analysis_final"
)

ALL_SLIDE_STATISTICS = (
    BIOLOGICAL_ANALYSIS_DIR /
    "ALL_SLIDE_STATISTICS.csv"
)

CELLS_DIR = (
    BIOLOGICAL_ANALYSIS_DIR /
    "cells"
)


# -----------------------------------------------------------------------------
# NEUTROPHIL EXPERT
# -----------------------------------------------------------------------------

MODEL_CHECKPOINT = Path(
    "/vol/firstbatch/neutrophil_expert_finetuned.pth"
)

MODEL_NAME = (
    "hf_hub:1aurent/vit_base_patch14_224.dinobloom"
)


# -----------------------------------------------------------------------------
# OUTPUT
# -----------------------------------------------------------------------------

OUTPUT_DIR = Path(
    "/vol/firstbatch/neutrophil_subtype_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CELL_OUTPUT_DIR = (
    OUTPUT_DIR /
    "cells"
)

CELL_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


SLIDE_OUTPUT_DIR = (
    OUTPUT_DIR /
    "slide_statistics"
)

SLIDE_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# -----------------------------------------------------------------------------
# MODEL / IMAGE PARAMETERS
# -----------------------------------------------------------------------------

IMG_SIZE = 224

BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# -----------------------------------------------------------------------------
# EXACT CLASS ORDER USED DURING EXPERT TRAINING
# -----------------------------------------------------------------------------

CLASSES = [
    "Immature_Neu",
    "Mature_Neu_SiglecF+",
    "Mature_Neu_SiglecF-",
]


# -----------------------------------------------------------------------------
# TTA
# -----------------------------------------------------------------------------

USE_TTA = True


# -----------------------------------------------------------------------------
# RESUME
# -----------------------------------------------------------------------------

SKIP_EXISTING = True


# =============================================================================
# 1. MODEL PREPROCESSING
# =============================================================================

def smart_pad(img):
    """
    Exact preprocessing logic used during expert training.

    Input:
        PIL.Image

    Output:
        224 x 224 PIL.Image
    """

    delta_w = IMG_SIZE - img.size[0]
    delta_h = IMG_SIZE - img.size[1]

    if delta_w <= 0 and delta_h <= 0:
        return img.resize(
            (IMG_SIZE, IMG_SIZE),
            Image.BILINEAR
        )

    pad_w = max(
        0,
        delta_w // 2
    )

    pad_h = max(
        0,
        delta_h // 2
    )

    return TF.pad(
        img,
        (
            pad_w,
            pad_h,
            delta_w - pad_w,
            delta_h - pad_h,
        ),
        padding_mode="reflect"
    )


PREPROCESS = transforms.Compose([
    transforms.Lambda(smart_pad),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ],
    ),
])


# =============================================================================
# 2. MODEL
# =============================================================================

def create_model():

    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=len(CLASSES),
        img_size=IMG_SIZE,
        global_pool="avg",
    )

    return model


def load_model():

    print("\n" + "=" * 80)
    print("LOADING NEUTROPHIL EXPERT")
    print("=" * 80)

    if not MODEL_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{MODEL_CHECKPOINT}"
        )

    model = create_model()

    checkpoint = torch.load(
        MODEL_CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint,
        strict=True,
    )

    model.to(DEVICE)

    model.eval()

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Checkpoint: {MODEL_CHECKPOINT}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Classes: {CLASSES}"
    )

    print("=" * 80)

    return model


# =============================================================================
# 3. CZI IMAGE LOADING
# =============================================================================

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
                (1, 2, 0)
            )

        elif img.shape[-1] not in [1, 2, 3, 4]:

            raise RuntimeError(
                f"Unexpected image shape: {img.shape}"
            )

    elif img.ndim == 2:

        img = np.stack(
            [
                img,
                img,
                img
            ],
            axis=-1
        )

    else:

        raise RuntimeError(
            f"Unexpected CZI dimensions: {img.shape}"
        )

    if img.shape[-1] == 1:

        img = np.repeat(
            img,
            3,
            axis=-1
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

            img = img.astype(
                np.uint8
            )

    return img


# =============================================================================
# 4. SLIDE NAME NORMALIZATION
# =============================================================================

def normalize_slide_name(name):

    name = str(name)

    name = Path(name).stem

    name = name.replace(
        " ",
        "_"
    )

    return name


# =============================================================================
# 5. FIND CZI SLIDE
# =============================================================================

def build_czi_index():

    print(
        "\nBuilding CZI slide index..."
    )

    index = {}

    for filepath in SLIDES_DIR.rglob("*.czi"):

        normalized = normalize_slide_name(
            filepath.stem
        )

        index[normalized] = filepath

    print(
        f"Found {len(index)} CZI slides."
    )

    return index


# =============================================================================
# 6. LOAD ALL SLIDE STATISTICS
# =============================================================================

def load_slide_statistics():

    if not ALL_SLIDE_STATISTICS.exists():

        raise FileNotFoundError(
            f"Missing:\n"
            f"{ALL_SLIDE_STATISTICS}"
        )

    df = pd.read_csv(
        ALL_SLIDE_STATISTICS
    )

    df["Slide_Normalized"] = (
        df["Slide"]
        .astype(str)
        .map(normalize_slide_name)
    )

    print(
        f"Loaded {len(df)} slide summaries."
    )

    return df


# =============================================================================
# 7. LOAD NEUTROPHIL COORDINATES
# =============================================================================

def load_neutrophil_cells(
    slide_name
):

    normalized = normalize_slide_name(
        slide_name
    )

    cells_path = (
        CELLS_DIR /
        f"{normalized}_cells.csv"
    )

    if not cells_path.exists():

        raise FileNotFoundError(
            f"Cell table not found:\n"
            f"{cells_path}"
        )

    df = pd.read_csv(
        cells_path
    )

    # -------------------------------------------------------------------------
    # We use the ORIGINAL global classifier result.
    # Only cells classified as Neutrophil are sent to the expert.
    # -------------------------------------------------------------------------

    if "Class" not in df.columns:

        raise RuntimeError(
            f"'Class' column missing in:\n"
            f"{cells_path}"
        )

    neutrophils = df[
        df["Class"].astype(str)
        == "Neutrophil"
    ].copy()

    required = [
        "Centroid_X",
        "Centroid_Y"
    ]

    missing = [
        c for c in required
        if c not in neutrophils.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing coordinate columns "
            f"{missing} in:\n"
            f"{cells_path}"
        )

    neutrophils[
        "Centroid_X"
    ] = pd.to_numeric(
        neutrophils["Centroid_X"],
        errors="coerce"
    )

    neutrophils[
        "Centroid_Y"
    ] = pd.to_numeric(
        neutrophils["Centroid_Y"],
        errors="coerce"
    )

    neutrophils = neutrophils.dropna(
        subset=[
            "Centroid_X",
            "Centroid_Y"
        ]
    ).reset_index(
        drop=True
    )

    return neutrophils


# =============================================================================
# 8. EXTRACT 224 x 224 CROP
# =============================================================================

def extract_crop(
    img,
    cx,
    cy
):

    H, W = img.shape[:2]

    half = IMG_SIZE // 2

    cx = int(round(cx))
    cy = int(round(cy))

    x1 = cx - half
    x2 = cx + half

    y1 = cy - half
    y2 = cy + half

    pad_left = max(
        0,
        -x1
    )

    pad_right = max(
        0,
        x2 - W
    )

    pad_top = max(
        0,
        -y1
    )

    pad_bottom = max(
        0,
        y2 - H
    )

    crop = img[
        max(0, y1):min(H, y2),
        max(0, x1):min(W, x2)
    ]

    if (
        pad_left
        or pad_right
        or pad_top
        or pad_bottom
    ):

        crop = cv2_copy_make_border(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right
        )

    return crop


def cv2_copy_make_border(
    crop,
    pad_top,
    pad_bottom,
    pad_left,
    pad_right
):

    import cv2

    return cv2.copyMakeBorder(
        crop,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_REFLECT
    )


# =============================================================================
# 9. TTA PREDICTION
# =============================================================================

def predict_batch(
    model,
    images
):

    tensors = []

    for crop in images:

        pil = Image.fromarray(
            crop
        )

        tensor = PREPROCESS(
            pil
        )

        tensors.append(
            tensor
        )

    batch = torch.stack(
        tensors
    ).to(
        DEVICE,
        non_blocking=True
    )

    with torch.no_grad():

        if not USE_TTA:

            logits = model(
                batch
            )

        else:

            logits_0 = model(
                batch
            )

            logits_90 = model(
                torch.rot90(
                    batch,
                    1,
                    [2, 3]
                )
            )

            logits_180 = model(
                torch.rot90(
                    batch,
                    2,
                    [2, 3]
                )
            )

            logits_270 = model(
                torch.rot90(
                    batch,
                    3,
                    [2, 3]
                )
            )

            logits = (
                logits_0
                + logits_90
                + logits_180
                + logits_270
            ) / 4.0

        probabilities = torch.softmax(
            logits,
            dim=1
        )

    result = (
        probabilities
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    del batch
    del tensors
    del logits

    return result


# =============================================================================
# 10. ENTROPY
# =============================================================================

def calculate_entropy(probabilities):

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )

    probabilities = np.clip(
        probabilities,
        1e-12,
        1.0
    )

    return float(
        -np.sum(
            probabilities
            * np.log(
                probabilities
            )
        )
    )


# =============================================================================
# 11. CLASSIFY ALL NEUTROPHILS ON ONE SLIDE
# =============================================================================

def classify_neutrophils(
    model,
    slide_name,
    slide_path,
    neutrophil_df
):

    print(
        "\n" + "-" * 80
    )

    print(
        f"PROCESSING SLIDE: "
        f"{slide_name}"
    )

    print(
        f"Neutrophils: "
        f"{len(neutrophil_df):,}"
    )

    print(
        f"CZI: {slide_path}"
    )

    print(
        "-" * 80
    )

    start_time = time.time()

    print(
        "Loading CZI..."
    )

    img = load_czi_image(
        slide_path
    )

    H, W = img.shape[:2]

    print(
        f"Image size: "
        f"{W:,} × {H:,}"
    )

    all_records = []

    total = len(
        neutrophil_df
    )

    for start in tqdm(
        range(
            0,
            total,
            BATCH_SIZE
        ),
        desc=f"Expert {slide_name}",
        unit="batch",
        dynamic_ncols=True
    ):

        batch_df = neutrophil_df.iloc[
            start:start + BATCH_SIZE
        ]

        crops = []

        valid_rows = []

        for _, row in batch_df.iterrows():

            cx = row["Centroid_X"]
            cy = row["Centroid_Y"]

            # --------------------------------------------------------------
            # Coordinate sanity check
            # --------------------------------------------------------------

            if (
                cx < 0
                or cy < 0
                or cx >= W
                or cy >= H
            ):
                continue

            crop = extract_crop(
                img,
                cx,
                cy
            )

            # Make absolutely sure crop is 224 x 224.
            if crop.shape[:2] != (
                IMG_SIZE,
                IMG_SIZE
            ):
                continue

            crops.append(
                crop
            )

            valid_rows.append(
                row
            )

        if not crops:
            continue

        probabilities = predict_batch(
            model,
            crops
        )

        for i, row in enumerate(
            valid_rows
        ):

            probs = probabilities[i]

            predicted_idx = int(
                np.argmax(
                    probs
                )
            )

            predicted_class = (
                CLASSES[
                    predicted_idx
                ]
            )

            confidence = float(
                probs[
                    predicted_idx
                ]
            )

            entropy = (
                calculate_entropy(
                    probs
                )
            )

            record = {
                "Slide": slide_name,

                "Original_Cell_ID":
                    row.get(
                        "Cell_ID",
                        np.nan
                    ),

                "Centroid_X":
                    row["Centroid_X"],

                "Centroid_Y":
                    row["Centroid_Y"],

                "Original_Class":
                    row["Class"],

                "Expert_Predicted_Class":
                    predicted_class,

                "Expert_Confidence":
                    confidence,

                "Expert_Entropy":
                    entropy,
            }

            for class_idx, class_name in enumerate(
                CLASSES
            ):

                record[
                    f"Prob_{class_name}"
                ] = float(
                    probs[class_idx]
                )

            all_records.append(
                record
            )

        del crops
        del probabilities

    del img

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = pd.DataFrame(
        all_records
    )

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"Finished {slide_name} "
        f"in {elapsed / 60:.1f} min"
    )

    print(
        f"Successfully classified: "
        f"{len(result):,}"
    )

    return result


# =============================================================================
# 12. SLIDE SUMMARY
# =============================================================================

def create_slide_summary(
    df,
    slide_metadata
):

    total = len(df)

    result = {
        "Slide":
            slide_metadata["Slide"],

        "Treatment":
            slide_metadata["Treatment"],

        "Timepoint":
            slide_metadata["Timepoint"],

        "Experimental_Group":
            slide_metadata[
                "Experimental_Group"
            ],

        "Original_Neutrophils":
            total,
    }

    # -------------------------------------------------------------------------
    # Counts
    # -------------------------------------------------------------------------

    counts = (
        df[
            "Expert_Predicted_Class"
        ]
        .value_counts()
    )

    for cls in CLASSES:

        result[
            f"Count_{cls}"
        ] = int(
            counts.get(
                cls,
                0
            )
        )

    # -------------------------------------------------------------------------
    # Percentages
    # -------------------------------------------------------------------------

    for cls in CLASSES:

        count = result[
            f"Count_{cls}"
        ]

        result[
            f"Share_{cls}_of_Neutrophils_pct"
        ] = (
            100.0
            * count
            / total
            if total > 0
            else 0.0
        )

    # -------------------------------------------------------------------------
    # Confidence
    # -------------------------------------------------------------------------

    if total > 0:

        result[
            "Mean_Expert_Confidence"
        ] = float(
            df[
                "Expert_Confidence"
            ].mean()
        )

        result[
            "Median_Expert_Confidence"
        ] = float(
            df[
                "Expert_Confidence"
            ].median()
        )

        result[
            "Mean_Expert_Entropy"
        ] = float(
            df[
                "Expert_Entropy"
            ].mean()
        )

        result[
            "Median_Expert_Entropy"
        ] = float(
            df[
                "Expert_Entropy"
            ].median()
        )

        result[
            "Low_Confidence_Share_pct"
        ] = float(
            (
                df[
                    "Expert_Confidence"
                ] < 0.60
            ).mean()
            * 100.0
        )

        result[
            "High_Confidence_Share_pct"
        ] = float(
            (
                df[
                    "Expert_Confidence"
                ] >= 0.90
            ).mean()
            * 100.0
        )

    else:

        result[
            "Mean_Expert_Confidence"
        ] = np.nan

        result[
            "Median_Expert_Confidence"
        ] = np.nan

        result[
            "Mean_Expert_Entropy"
        ] = np.nan

        result[
            "Median_Expert_Entropy"
        ] = np.nan

        result[
            "Low_Confidence_Share_pct"
        ] = np.nan

        result[
            "High_Confidence_Share_pct"
        ] = np.nan

    # -------------------------------------------------------------------------
    # Original neutrophil count from ALL_SLIDE_STATISTICS
    # -------------------------------------------------------------------------

    if (
        "Count_Neutrophil"
        in slide_metadata
    ):

        result[
            "Original_Neutrophil_Count_From_Summary"
        ] = slide_metadata[
            "Count_Neutrophil"
        ]

    return pd.DataFrame(
        [result]
    )


# =============================================================================
# 13. MAIN
# =============================================================================

def main():

    print(
        "\n" + "=" * 80
    )

    print(
        "NEUTROPHIL SUBTYPE ANALYSIS"
    )

    print(
        "=" * 80
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Slides: {SLIDES_DIR}"
    )

    print(
        f"Cell tables: {CELLS_DIR}"
    )

    print(
        f"Model: {MODEL_CHECKPOINT}"
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )

    print(
        "=" * 80
    )


    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    slide_stats = (
        load_slide_statistics()
    )

    czi_index = (
        build_czi_index()
    )

    model = load_model()


    # -------------------------------------------------------------------------
    # Track results
    # -------------------------------------------------------------------------

    all_slide_summaries = []

    failed = []

    processed = 0

    skipped = 0


    # -------------------------------------------------------------------------
    # Process slides
    # -------------------------------------------------------------------------

    for slide_idx, slide_metadata in tqdm(
        slide_stats.iterrows(),
        total=len(slide_stats),
        desc="Slides",
        unit="slide"
    ):

        slide_name = str(
            slide_metadata[
                "Slide"
            ]
        )

        normalized = (
            normalize_slide_name(
                slide_name
            )
        )


        # ---------------------------------------------------------------------
        # Output paths
        # ---------------------------------------------------------------------

        cell_output = (
            CELL_OUTPUT_DIR /
            f"{normalized}_neutrophil_subtypes.csv"
        )

        summary_output = (
            SLIDE_OUTPUT_DIR /
            f"{normalized}_neutrophil_summary.csv"
        )


        # ---------------------------------------------------------------------
        # Resume
        # ---------------------------------------------------------------------

        if (
            SKIP_EXISTING
            and cell_output.exists()
            and summary_output.exists()
        ):

            print(
                f"\nSKIPPING: "
                f"{slide_name}"
            )

            try:

                existing_summary = (
                    pd.read_csv(
                        summary_output
                    )
                )

                all_slide_summaries.append(
                    existing_summary
                )

                skipped += 1

            except Exception as e:

                print(
                    f"Could not load existing "
                    f"summary: {e}"
                )

            continue


        # ---------------------------------------------------------------------
        # Find CZI
        # ---------------------------------------------------------------------

        if normalized not in czi_index:

            print(
                f"\nWARNING: CZI not found "
                f"for {slide_name}"
            )

            failed.append({
                "Slide":
                    slide_name,

                "Error":
                    "CZI file not found"
            })

            continue


        slide_path = (
            czi_index[
                normalized
            ]
        )


        # ---------------------------------------------------------------------
        # Load neutrophil coordinates
        # ---------------------------------------------------------------------

        try:

            neutrophils = (
                load_neutrophil_cells(
                    slide_name
                )
            )

        except Exception as e:

            print(
                f"\nFAILED loading "
                f"neutrophils: "
                f"{slide_name}"
            )

            print(
                repr(e)
            )

            failed.append({
                "Slide":
                    slide_name,

                "Error":
                    repr(e)
            })

            continue


        # ---------------------------------------------------------------------
        # No neutrophils
        # ---------------------------------------------------------------------

        if len(neutrophils) == 0:

            print(
                f"\nNo neutrophils found "
                f"on {slide_name}"
            )

            empty = pd.DataFrame(
                columns=[
                    "Slide",
                    "Original_Cell_ID",
                    "Centroid_X",
                    "Centroid_Y",
                    "Original_Class",
                    "Expert_Predicted_Class",
                    "Expert_Confidence",
                    "Expert_Entropy",
                ]
                + [
                    f"Prob_{cls}"
                    for cls in CLASSES
                ]
            )

            empty.to_csv(
                cell_output,
                index=False
            )

            metadata_dict = (
                slide_metadata.to_dict()
            )

            summary = create_slide_summary(
                empty,
                metadata_dict
            )

            summary.to_csv(
                summary_output,
                index=False
            )

            all_slide_summaries.append(
                summary
            )

            processed += 1

            continue


        # ---------------------------------------------------------------------
        # Classify
        # ---------------------------------------------------------------------

        try:

            cell_predictions = (
                classify_neutrophils(
                    model,
                    slide_name,
                    slide_path,
                    neutrophils
                )
            )

            # ---------------------------------------------------------------
            # Save cell-level results
            # ---------------------------------------------------------------

            cell_predictions.to_csv(
                cell_output,
                index=False
            )

            print(
                f"Saved cell predictions:\n"
                f"{cell_output}"
            )


            # ---------------------------------------------------------------
            # Slide summary
            # ---------------------------------------------------------------

            metadata_dict = (
                slide_metadata.to_dict()
            )

            summary = create_slide_summary(
                cell_predictions,
                metadata_dict
            )

            summary.to_csv(
                summary_output,
                index=False
            )

            print(
                f"Saved slide summary:\n"
                f"{summary_output}"
            )


            # ---------------------------------------------------------------
            # Print distribution
            # ---------------------------------------------------------------

            print(
                "\nSubtype distribution:"
            )

            if len(cell_predictions) > 0:

                print(
                    cell_predictions[
                        "Expert_Predicted_Class"
                    ]
                    .value_counts()
                    .to_string()
                )

                print(
                    "\nMean confidence: "
                    f"{cell_predictions['Expert_Confidence'].mean():.4f}"
                )

            all_slide_summaries.append(
                summary
            )

            processed += 1


        except Exception as e:

            print(
                f"\nFAILED slide:"
            )

            print(
                slide_name
            )

            print(
                repr(e)
            )

            failed.append({
                "Slide":
                    slide_name,

                "Error":
                    repr(e)
            })

            continue


    # =========================================================================
    # 14. COMBINE RESULTS
    # =========================================================================

    if all_slide_summaries:

        combined = pd.concat(
            all_slide_summaries,
            ignore_index=True
        )

        combined_path = (
            OUTPUT_DIR /
            "ALL_NEUTROPHIL_SUBTYPE_STATISTICS.csv"
        )

        combined.to_csv(
            combined_path,
            index=False
        )

        print(
            "\nSaved combined slide statistics:"
        )

        print(
            combined_path
        )


        # ---------------------------------------------------------------------
        # Treatment × timepoint summary
        # ---------------------------------------------------------------------

        numeric_cols = (
            combined
            .select_dtypes(
                include=np.number
            )
            .columns
        )

        group_cols = [
            "Treatment",
            "Timepoint",
            "Experimental_Group"
        ]

        group_summary = (
            combined
            .groupby(
                group_cols,
                dropna=False
            )[numeric_cols]
            .agg(
                [
                    "mean",
                    "std",
                    "median",
                    "count"
                ]
            )
            .reset_index()
        )

        group_summary.columns = [
            (
                f"{a}_{b}"
                if b
                else str(a)
            )
            if isinstance(
                (a, b),
                tuple
            )
            else str(a)

            for a, b in [
                c if isinstance(
                    c,
                    tuple
                )
                else (c, "")
                for c in group_summary.columns
            ]
        ]

        group_path = (
            OUTPUT_DIR /
            "NEUTROPHIL_SUBTYPE_GROUP_STATISTICS.csv"
        )

        group_summary.to_csv(
            group_path,
            index=False
        )

        print(
            "Saved group-level statistics:"
        )

        print(
            group_path
        )


    # =========================================================================
    # 15. FAILED SLIDES
    # =========================================================================

    if failed:

        failed_df = pd.DataFrame(
            failed
        )

        failed_path = (
            OUTPUT_DIR /
            "FAILED_SLIDES.csv"
        )

        failed_df.to_csv(
            failed_path,
            index=False
        )

        print(
            "\nFailed slides saved:"
        )

        print(
            failed_path
        )


    # =========================================================================
    # 16. FINAL REPORT
    # =========================================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "NEUTROPHIL SUBTYPE ANALYSIS FINISHED"
    )

    print(
        "=" * 80
    )

    print(
        f"Processed: "
        f"{processed}"
    )

    print(
        f"Skipped: "
        f"{skipped}"
    )

    print(
        f"Failed: "
        f"{len(failed)}"
    )

    print(
        f"Output directory:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()