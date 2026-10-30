"""Lab handover of the BAL production inference pipeline. See IMPLEMENTATION_NOTES.md.

Importing this module never loads weights, downloads models or processes images.
"""
from __future__ import annotations

import csv
import dataclasses
import gc
import hashlib
import importlib.metadata as imd
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

VERSION = "1.0.3-lab"
CLASSES = ("Macrophage", "Macrophage_with_particles", "Neutrophil", "Erythrocyte",
           "Lymphocyte", "Monocyte", "Eosinophil", "Junk")
EXPERT_CLASSES = ("Immature_Neu", "Mature_Neu_SiglecF+", "Mature_Neu_SiglecF-")
EXPERT_SHORT = dict(zip(EXPERT_CLASSES, ("Immature Neu", "Mature Neu F+", "Mature Neu F-")))
WBC_CLASSES = tuple(c for c in CLASSES if c not in ("Erythrocyte", "Junk"))
SHORT = dict(zip(CLASSES, ("Mac", "Mac+p", "Neu", "Ery", "Lym", "Mon", "Eos", "Junk")))
COLORS = dict(zip(CLASSES, ((62, 132, 110), (161, 76, 143), (65, 113, 154), (185, 112, 117),
                          (129, 111, 171), (194, 151, 70), (79, 155, 165), (135, 141, 146))))
CELL_COLUMNS = ["Slide", "Animal_ID", "Treatment", "Timepoint", "Cell_ID", "Class",
                "Model_Predicted_Class", "Model_Confidence", "Particle_Rule_Applied",
                "Dark_Structure_Count", "Nanotubes_Count", "Centroid_Y", "Centroid_X",
                "BBox_MinY", "BBox_MinX", "BBox_MaxY", "BBox_MaxX", "Segmented_Area",
                "Touches_Tile_Border", "Expert_Class", "Expert_Confidence", "Expert_Entropy"]
CELL_COLUMNS += [f"Prob_{c}" for c in CLASSES] + [f"Expert_Prob_{c}" for c in EXPERT_CLASSES]


@dataclass(frozen=True)
class Config:
    classifier_checkpoint: str
    expert_checkpoint: str
    cellpose_checkpoint: str
    output_dir: str
    enable_expert: bool = True
    device: str = "auto"  # Cellpose: CUDA, then Apple MPS, then CPU
    classifier_device: str = "same"  # same, auto, cpu, cuda or mps
    tile_size: int = 2500
    padding: int = 256
    batch_size: int = 8
    cellpose_batch_size: int = 2
    min_area: int = 150
    flow_threshold: float = 0.9
    cellprob_threshold: float = 0.0
    nms_iou: float = 0.5
    rgb_dark_threshold: int = 75
    min_peak_distance: int = 2
    large_dark_area: int = 2000
    particle_class_threshold: int = 3
    min_saturation: int = 18
    min_contrast: int = 20
    save_annotated_tiff: bool = True
    save_preview_pairs: int = 3
    contour_width: int = 1
    expert_tta: bool = True
    cellpose_bfloat16: bool = False

    def __post_init__(self):
        if not str(self.output_dir).strip():
            raise ValueError("Choose an explicit results directory.")
        for name in ("classifier_checkpoint","expert_checkpoint","cellpose_checkpoint","output_dir"):
            value=str(getattr(self,name))
            if name=="cellpose_checkpoint" and value=="cpsam":
                continue
            if value:
                object.__setattr__(self,name,str(Path(value).expanduser().resolve()))

    def validate(self):
        if not self.cellpose_checkpoint:
            raise ValueError("Use cellpose_checkpoint='cpsam' for automatic library caching, or an explicit local file path.")
        if self.device not in ("auto", "cpu", "cuda", "mps"):
            raise ValueError("DEVICE must be auto, cpu, cuda or mps.")
        if self.classifier_device not in ("same", "auto", "cpu", "cuda", "mps"):
            raise ValueError("CLASSIFIER_DEVICE must be same, auto, cpu, cuda or mps.")
        if self.tile_size < 224 or self.padding < 0 or self.min_area < 1:
            raise ValueError("Tile size must be >=224; padding >=0; minimum area >=1.")
        for name in ("batch_size", "cellpose_batch_size", "contour_width", "min_peak_distance",
                     "large_dark_area", "particle_class_threshold"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if not 0 <= self.nms_iou <= 1 or not 0 <= self.rgb_dark_threshold <= 255:
            raise ValueError("Invalid IoU or RGB dark threshold.")
        if self.save_preview_pairs < 0:
            raise ValueError("save_preview_pairs cannot be negative.")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve_cellpose_checkpoint(value):
    """Resolve the pinned library's cpsam cache only when processing starts."""
    if value != "cpsam":
        path=Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Explicit Cellpose weights not found: {path}. No automatic fallback was used.")
        return path
    from cellpose import models as cp_models
    if imd.version("cellpose") != "4.0.6":
        raise RuntimeError("This notebook pins cellpose==4.0.6. Install requirements.txt to use its original cpsam cache API.")
    print("Checking the Cellpose cpsam cache; first use downloads the model if needed...",flush=True)
    try:
        path=Path(cp_models.cache_CPSAM_model_path()).expanduser().resolve()
    except Exception as error:
        raise RuntimeError("Cellpose could not obtain cpsam. The first download needs internet access. "
                           "Alternatively, set CELLPOSE_MODEL to an existing local cpsam file.") from error
    if not path.is_file() or path.stat().st_size==0:
        raise RuntimeError("Cellpose returned missing/empty cpsam weights. Check its cache or supply a local file.")
    return path


def json_write(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False,
                              default=lambda x: int(x) if isinstance(x, np.integer) else str(x)),
                   encoding="utf-8")
    tmp.replace(path)


def safe_name(path):
    return re.sub(r"[^\w.-]+", "_", Path(path).stem, flags=re.UNICODE).strip("._") or "slide"


def versions():
    packages = ("numpy", "pandas", "torch", "torchvision", "timm", "cellpose", "czifile",
                "opencv-python-headless", "scikit-image", "tifffile", "imagecodecs", "Pillow")
    out = {"Python": sys.version, "Platform": platform.platform(), "Pipeline": VERSION}
    for p in packages:
        try:
            out[p] = imd.version(p)
        except imd.PackageNotFoundError:
            out[p] = "not installed"
    return out


def select_device(requested, torch_module=None):
    """Select an available backend without loading models or changing memory guards."""
    if torch_module is None:
        import torch as torch_module
    cuda=bool(torch_module.cuda.is_available())
    mps_backend=getattr(getattr(torch_module,"backends",None),"mps",None)
    mps=bool(mps_backend is not None and mps_backend.is_available())
    if requested=="auto":
        return "cuda" if cuda else "mps" if mps else "cpu"
    if requested=="cuda" and not cuda:
        raise RuntimeError("CUDA was requested but is unavailable. Check the NVIDIA build/driver or set DEVICE='cpu'.")
    if requested=="mps" and not mps:
        raise RuntimeError("MPS was requested but is unavailable. Use a supported Mac/macOS with an MPS-enabled PyTorch build, or select cpu.")
    if requested not in ("cpu","cuda","mps"):
        raise ValueError(f"Unsupported device: {requested}")
    return requested


def execution_plan(config, torch_module=None):
    cellpose=select_device(config.device,torch_module)
    classifier=cellpose if config.classifier_device=="same" else select_device(config.classifier_device,torch_module)
    return {"cellpose_device":cellpose,"classifier_device":classifier,
            "cellpose_dtype":"bfloat16" if config.cellpose_bfloat16 and cellpose!="mps" else "float32",
            "classifier_dtype":"float32",
            "pytorch_mps_fallback_env":os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK","unset")}


def discover_slides(folder):
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Input folder not found: {folder}")
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".czi", ".tif", ".tiff"))


def load_samples(paths, metadata_csv=None):
    """Map explicit filenames to metadata, without historical filename guessing."""
    paths = [Path(p).expanduser().resolve() for p in paths]
    if not paths:
        raise ValueError("No slides selected. Copy a CZI/RGB TIFF into input_slides, or set SLIDE_FILE in the notebook.")
    if len(set(paths)) != len(paths):
        raise ValueError("The same input file was selected twice.")
    if len({safe_name(p).casefold() for p in paths}) != len(paths):
        raise ValueError("Slide names collide after normalisation. Use distinct filenames.")
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"Slide not found: {p}")
    lookup = {}
    if metadata_csv and not Path(metadata_csv).expanduser().is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}. Set METADATA_CSV=None to analyse slides without animal grouping.")
    if metadata_csv:
        metadata_csv=Path(metadata_csv).expanduser()
        meta = pd.read_csv(metadata_csv, dtype=str, keep_default_na=False)
        required = ("filename", "animal_id", "treatment", "timepoint")
        if not set(required).issubset(meta.columns):
            raise ValueError(f"Metadata needs these columns: {required}")
        if meta.filename.duplicated().any():
            raise ValueError("Metadata contains duplicate filenames.")
        lookup = meta.set_index("filename").to_dict("index")
        missing = [p.name for p in paths if p.name not in lookup]
        if missing:
            raise ValueError(f"Add metadata rows for the selected files: {missing}")
    result = []
    for p in paths:
        m = lookup.get(p.name, {})
        result.append({"path": str(p), "filename": p.name, "animal_id": m.get("animal_id", "").strip(),
                       "treatment": m.get("treatment", "").strip(), "timepoint": m.get("timepoint", "").strip()})
    seen = {}
    for sample in result:
        animal = sample["animal_id"]
        if animal:
            condition = sample["treatment"], sample["timepoint"]
            if animal in seen and seen[animal] != condition:
                raise ValueError(f"Animal {animal} has inconsistent treatment/timepoint metadata.")
            seen[animal] = condition
    return result


def axis_plan(shape, axes):
    """Select only singleton non-image axes, then return a view in RGB YXC order."""
    if len(shape) != len(axes) or len(set(axes)) != len(axes):
        raise ValueError(f"Unsupported image axes: {axes}, shape={shape}")
    if "Y" not in axes or "X" not in axes:
        raise ValueError("Expected spatial Y and X axes.")
    candidates = [i for i, (a, n) in enumerate(zip(axes, shape)) if a in ("C", "S", "0") and n == 3]
    if len(candidates) != 1:
        raise ValueError(f"Expected one RGB axis of length 3. Found axes {axes}, shape {shape}. "
                         "Export a single 8-bit RGB plane from ZEN; do not use a fluorescence channel stack.")
    color_i = candidates[0]
    keep = [axes.index("Y"), axes.index("X"), color_i]
    selection = []
    kept_indices = []
    for i, n in enumerate(shape):
        if i in keep:
            selection.append(slice(None)); kept_indices.append(i)
        elif n == 1:
            selection.append(0)
        else:
            raise ValueError(f"Axis {axes[i]} has {n} planes. This notebook accepts one scene/time/Z plane. "
                             "Export the intended RGB plane explicitly instead of silently choosing the first.")
    transpose = tuple(kept_indices.index(i) for i in keep)
    return tuple(selection), transpose


def czi_pixel_size(metadata_xml):
    sizes = {}
    try:
        root = ET.fromstring(metadata_xml or "")
        for node in root.iter():
            if node.tag.split("}")[-1] == "Distance" and node.attrib.get("Id") in ("X", "Y"):
                value = next((c.text for c in node if c.tag.split("}")[-1] == "Value"), None)
                if value is not None:
                    sizes[node.attrib["Id"]] = float(value) * 1e6
    except (ET.ParseError, ValueError, TypeError):
        pass
    return sizes


def inspect_slide(path):
    """Header-only inspection: no pixels or weights are loaded."""
    path = Path(path)
    if path.suffix.lower() == ".czi":
        import czifile
        if imd.version("czifile") != "2019.7.2.3":
            raise RuntimeError("Install czifile==2019.7.2.3 from requirements.txt; newer APIs differ.")
        with czifile.CziFile(str(path)) as reader:
            shape, axes, dtype = tuple(reader.shape), reader.axes, np.dtype(reader.dtype)
            pixel_size = czi_pixel_size(reader.metadata())
    elif path.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        with tifffile.TiffFile(path) as reader:
            if len(reader.series) != 1:
                raise ValueError("TIFF must contain a single RGB image series.")
            series = reader.series[0]
            shape, axes, dtype = tuple(series.shape), series.axes, np.dtype(series.dtype)
            if len(series.levels) > 1:
                raise ValueError("Export a single full-resolution RGB TIFF rather than a pyramid.")
            pixel_size = {}
            page=series.pages[0]
            unit=page.tags.get("ResolutionUnit")
            unit_um={2:25400.0,3:10000.0}.get(int(unit.value)) if unit is not None else None
            if unit_um:
                for axis,tag in (("X","XResolution"),("Y","YResolution")):
                    value=page.tags.get(tag)
                    if value is not None:
                        numerator,denominator=value.value
                        if numerator>0 and denominator>0:
                            pixel_size[axis]=unit_um*denominator/numerator
    else:
        raise ValueError("Supported inputs are .czi and single-plane RGB .tif/.tiff.")
    selection, transpose = axis_plan(shape, axes)
    if dtype != np.uint8:
        raise ValueError(f"{path.name} is {dtype}. Training used 8-bit RGB crops. "
                         "Export an 8-bit RGB image with the validated display settings; no automatic intensity rescaling is applied.")
    return {"filename": path.name, "width": int(shape[axes.index("X")]),
            "height": int(shape[axes.index("Y")]), "axes": axes, "shape": list(shape),
            "dtype": str(dtype), "raw_bytes": int(math.prod(shape) * dtype.itemsize),
            "pixel_size_um": pixel_size}


class SlideImage:
    """Whole-image decode to a temporary disk-backed array; inference uses bounded crops."""
    def __init__(self, path, cache_dir):
        self.path, self.cache_dir = Path(path), Path(cache_dir)
        self.info = inspect_slide(path)
        self.raw = self.rgb = None
        self.cache_path = self.cache_dir / "decoded_pixels.dat"

    def __enter__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.path.suffix.lower() == ".czi":
            import czifile
            with czifile.CziFile(str(self.path)) as reader:
                self.raw = reader.asarray(out=str(self.cache_path), max_workers=2)
        else:
            import tifffile
            self.raw = tifffile.imread(self.path, out=str(self.cache_path))
        selection, transpose = axis_plan(self.raw.shape, self.info["axes"])
        self.rgb = self.raw[selection].transpose(transpose)
        return self

    def __exit__(self, *exc):
        self.rgb = None
        if self.raw is not None:
            if hasattr(self.raw, "flush"):
                self.raw.flush()
            mmap = getattr(self.raw, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self.raw = None
        self.cache_path.unlink(missing_ok=True)


def preflight(config, samples):
    config.validate()
    required = ("torch", "torchvision", "timm", "cellpose", "cv2", "skimage", "tifffile", "czifile", "imagecodecs")
    missing = [n for n in required if importlib.util.find_spec(n) is None]
    if missing:
        raise RuntimeError(f"Missing packages: {', '.join(missing)}. Complete INSTALL.md and select the BAL analysis kernel.")
    checkpoints=[("Classifier", config.classifier_checkpoint)]
    if config.cellpose_checkpoint != "cpsam":
        checkpoints.append(("Cellpose-SAM", config.cellpose_checkpoint))
    else:
        print("Cellpose-SAM: built-in cpsam; downloaded/cached by Cellpose when processing starts.")
    for name, path in checkpoints + (
            [("Neutrophil expert", config.expert_checkpoint)] if config.enable_expert else []):
        if not Path(path).expanduser().is_file():
            raise FileNotFoundError(f"{name} weights not found: {path}. See models/PUT_WEIGHTS_HERE.md.")
    out = Path(config.output_dir).expanduser().resolve()
    for s in samples:
        if out == Path(s["path"]).parent or out in Path(s["path"]).parents:
            raise ValueError("Choose a results folder separate from the source slide folder.")
    rows = [inspect_slide(s["path"]) for s in samples]
    for row in rows:
        row["tiles"] = math.ceil(row["height"] / config.tile_size) * math.ceil(row["width"] / config.tile_size)
        row["decoded_GiB"] = row["raw_bytes"] / 2**30
    probe = out
    while not probe.exists():
        probe = probe.parent
    required_bytes = max(r["raw_bytes"] * 2 for r in rows)
    if config.save_annotated_tiff:
        required_bytes += sum(r["raw_bytes"] for r in rows)
    required_bytes = int(required_bytes * 1.25 + 2 * 2**30)
    free = shutil.disk_usage(probe).free
    if free < required_bytes:
        raise RuntimeError(f"Insufficient disk space: {free/2**30:.1f} GiB free; reserve about "
                           f"{required_bytes/2**30:.1f} GiB for decoding, annotations and tables.")
    plan=execution_plan(config)
    print(f"Cellpose: {plan['cellpose_device'].upper()} ({plan['cellpose_dtype']}); "
          f"classifiers: {plan['classifier_device'].upper()} (float32). Results folder: {out}")
    print(f"Disk free: {free/2**30:.1f} GiB; estimated reservation: {required_bytes/2**30:.1f} GiB.")
    if plan["cellpose_device"] == "cpu":
        print("CPU mode: start with the small preview. Full slides can take substantially longer.")
    if plan["cellpose_device"] == "mps":
        print("Apple MPS acceleration selected for Cellpose. Some mask-construction steps still run on CPU.")
    return pd.DataFrame(rows)


def bbox_iou(a, b):
    y1, x1 = max(a[0], b[0]), max(a[1], b[1])
    y2, x2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, y2-y1) * max(0, x2-x1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union else 0.0


def deduplicate(cells, threshold=0.5, grid_size=2500):
    """Area-first NMS on GLOBAL boxes, indexed by all touched spatial bins."""
    bins, kept = {}, []
    for cell in sorted(cells, key=lambda c: (-c["area"], c["bbox"][0], c["bbox"][1], c["source_id"])):
        y1, x1, y2, x2 = cell["bbox"]
        keys = [(y, x) for y in range(y1//grid_size, (y2-1)//grid_size+1)
                for x in range(x1//grid_size, (x2-1)//grid_size+1)]
        candidates = {i for k in keys for i in bins.get(k, [])}
        if any(bbox_iou(cell["bbox"], kept[i]["bbox"]) > threshold for i in candidates):
            continue
        idx = len(kept); kept.append(cell)
        for key in keys:
            bins.setdefault(key, []).append(idx)
    return sorted(kept, key=lambda c: c["source_id"])


def cells_from_masks(masks, y0, x0, min_area=150, source_start=0):
    import cv2
    from skimage.measure import regionprops
    cells = []
    h, w = masks.shape
    for prop in regionprops(masks):
        if prop.area < min_area:
            continue
        y1, x1, y2, x2 = prop.bbox
        contours, _ = cv2.findContours(prop.image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in contours:
            xy = contour.reshape(-1, 2).astype(np.int32)
            xy += np.array([x0+x1, y0+y1])
            polygons.append(xy.tolist())
        cells.append({"source_id": source_start+len(cells), "cy": y0+int(round(prop.centroid[0])),
                      "cx": x0+int(round(prop.centroid[1])), "bbox": [y0+y1,x0+x1,y0+y2,x0+x2],
                      "area": int(prop.area), "contours_xy": polygons,
                      "border": y1 == 0 or x1 == 0 or y2 == h or x2 == w})
    return cells


def crop_at(image, cy, cx, size=224):
    import cv2
    h, w = image.shape[:2]; half = size//2
    y1, x1, y2, x2 = cy-half, cx-half, cy+half, cx+half
    crop = np.asarray(image[max(0,y1):min(h,y2), max(0,x1):min(w,x2)])
    if not crop.size:
        raise ValueError("Cell centroid lies outside the image.")
    return cv2.copyMakeBorder(crop, max(0,-y1), max(0,y2-h), max(0,-x1), max(0,x2-w), cv2.BORDER_REFLECT)


def valid_crop(crop, config):
    import cv2
    sat = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)[:,:,1]
    if sat.mean() < config.min_saturation and sat.max() < 40:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return np.percentile(gray,95)-np.percentile(gray,5) >= config.min_contrast


def dark_structures(crop, config):
    """Production full-crop dark-structure proxy, including the large-blob rule."""
    import cv2
    from skimage.feature import peak_local_max
    dark = np.all(crop < config.rgb_dark_threshold, axis=-1)
    if not dark.any():
        return 0, []
    contours, _ = cv2.findContours(dark.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    inv = 255-cv2.GaussianBlur(cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY), (3,3), 0)
    count, locations = 0, []
    for contour in contours:
        mask = np.zeros(dark.shape,np.uint8)
        cv2.drawContours(mask,[contour],-1,1,cv2.FILLED)
        if cv2.contourArea(contour) >= config.large_dark_area:
            count += 1
            moments = cv2.moments(contour)
            if moments["m00"]:
                locations.append([int(moments["m01"]/moments["m00"]), int(moments["m10"]/moments["m00"])])
        else:
            peaks = peak_local_max(inv, min_distance=config.min_peak_distance, labels=mask.astype(bool))
            count += len(peaks); locations.extend(peaks.tolist())
    return count, locations


def make_record(cell, probs, crop, config, sample, cell_id):
    if not np.isfinite(probs).all() or not np.isclose(np.sum(probs),1,atol=1e-4) or np.min(probs)<0:
        raise RuntimeError("Classifier returned invalid probabilities. Check weights, precision and image inputs.")
    idx = int(np.argmax(probs)); original = CLASSES[idx]
    count, peaks = dark_structures(crop,config) if original in ("Macrophage", "Macrophage_with_particles", "Monocyte") else (0,[])
    final = "Macrophage_with_particles" if count >= config.particle_class_threshold else original
    y1,x1,y2,x2 = cell["bbox"]
    record = {"Slide":safe_name(sample["path"]), "Animal_ID":sample["animal_id"],
              "Treatment":sample["treatment"], "Timepoint":sample["timepoint"], "Cell_ID":cell_id,
              "Class":final, "Model_Predicted_Class":original, "Model_Confidence":float(probs[idx]),
              "Particle_Rule_Applied":final != original, "Dark_Structure_Count":int(count),
              "Nanotubes_Count":int(count), "Centroid_Y":cell["cy"], "Centroid_X":cell["cx"],
              "BBox_MinY":y1,"BBox_MinX":x1,"BBox_MaxY":y2,"BBox_MaxX":x2,
              "Segmented_Area":cell["area"],"Touches_Tile_Border":cell["border"],
              "Expert_Class":"", "Expert_Confidence":np.nan, "Expert_Entropy":np.nan}
    record.update({f"Prob_{c}":float(p) for c,p in zip(CLASSES,probs)})
    record.update({f"Expert_Prob_{c}":np.nan for c in EXPERT_CLASSES})
    return record, peaks


def annotation_text(record):
    """Compact final-class label; original scores remain in the exported cell table."""
    original, final = record["Model_Predicted_Class"], record["Class"]
    if final == "Neutrophil" and record.get("Expert_Class") in EXPERT_SHORT:
        return EXPERT_SHORT[record["Expert_Class"]]
    text = SHORT[final]
    if final != original:
        text += "*"
    count = int(record["Dark_Structure_Count"])
    if final == "Macrophage_with_particles" or count:
        text += f" | P:{count}"
    return text


def count_fields():
    return [f"Count_{c}" for c in CLASSES] + [f"Count_{c}" for c in EXPERT_CLASSES]


def add_proportions(row):
    total = row["Total_Detected_Cells"]
    wbc = sum(row[f"Count_{c}"] for c in WBC_CLASSES)
    mac = row["Count_Macrophage"]+row["Count_Macrophage_with_particles"]
    neu = row["Count_Neutrophil"]
    row.update(Total_WBC=wbc, Total_Macrophages=mac)
    pct = lambda n,d:100.0*n/d if d else np.nan
    for c in CLASSES:
        row[f"Share_{c}_AllCells_pct"] = pct(row[f"Count_{c}"],total)
    for c in WBC_CLASSES:
        row[f"BAL_Differential_{c}_pct"] = pct(row[f"Count_{c}"],wbc)
    row["BAL_Differential_Macrophages_Combined_pct"] = pct(mac,wbc)
    row["Particle_Positive_Macrophage_Share_pct"] = pct(row["Count_Macrophage_with_particles"],mac)
    row["Mean_Dark_Structures_per_Macrophage"] = row["Macrophage_Dark_Structure_Total"]/mac if mac else np.nan
    row["WBC_Share_of_AllCells_pct"] = pct(wbc,total)
    for c in EXPERT_CLASSES:
        row[f"Subtype_{c}_WithinNeutrophils_pct"] = pct(row[f"Count_{c}"],neu) if row["Expert_Enabled"] else np.nan
        row[f"Subtype_{c}_WithinWBC_pct"] = pct(row[f"Count_{c}"],wbc) if row["Expert_Enabled"] else np.nan
    return row


def summarise_cells(df, sample, expert_enabled):
    row = {"Slide":safe_name(sample["path"]), "Animal_ID":sample["animal_id"],
           "Treatment":sample["treatment"], "Timepoint":sample["timepoint"],
           "Total_Detected_Cells":len(df), "Expert_Enabled":bool(expert_enabled)}
    for c in CLASSES:
        row[f"Count_{c}"] = int((df.Class == c).sum())
    for c in EXPERT_CLASSES:
        row[f"Count_{c}"] = int((df.Expert_Class == c).sum())
    mac = df[df.Class.isin(("Macrophage","Macrophage_with_particles"))]
    row["Total_Dark_Structures"] = int(df.Dark_Structure_Count.sum())
    row["Macrophage_Dark_Structure_Total"] = int(mac.Dark_Structure_Count.sum())
    row["Particle_Rule_Changed_Cells"] = int(df.Particle_Rule_Applied.sum())
    row["Tile_Border_Cells"] = int(df.Touches_Tile_Border.sum())
    row["Mean_Original_Model_Confidence"] = float(df.Model_Confidence.mean())
    neu = df[df.Class == "Neutrophil"]
    row["Mean_Expert_Confidence"] = float(neu.Expert_Confidence.mean()) if expert_enabled else np.nan
    row["Expert_Below_0_60_pct"] = float((neu.Expert_Confidence < .6).mean()*100) if len(neu) and expert_enabled else np.nan
    assert sum(row[f"Count_{c}"] for c in CLASSES) == len(df)
    if expert_enabled:
        assert sum(row[f"Count_{c}"] for c in EXPERT_CLASSES) == row["Count_Neutrophil"]
    return add_proportions(row)


def aggregate_animals(slide_df):
    """Only explicit animal metadata is pooled; no inference from filenames."""
    if slide_df.empty or slide_df.Animal_ID.fillna("").eq("").any():
        return pd.DataFrame(), pd.DataFrame()
    if (slide_df.Scope != "full").any():
        raise ValueError("Partial runs cannot enter animal/group summaries.")
    sum_cols = ["Total_Detected_Cells", "Total_Dark_Structures", "Macrophage_Dark_Structure_Total",
                "Particle_Rule_Changed_Cells", "Tile_Border_Cells"] + count_fields()
    rows=[]
    for animal, part in slide_df.groupby("Animal_ID",sort=False):
        for col in ("Treatment","Timepoint","Expert_Enabled"):
            if part[col].nunique(dropna=False) != 1:
                raise ValueError(f"Inconsistent {col} for animal {animal}.")
        row={"Animal_ID":animal,"Treatment":part.Treatment.iloc[0],"Timepoint":part.Timepoint.iloc[0],
             "Expert_Enabled":bool(part.Expert_Enabled.iloc[0]),"Preparations":len(part)}
        row.update({c:int(part[c].sum()) for c in sum_cols})
        rows.append(add_proportions(row))
    animals=pd.DataFrame(rows)
    metrics=[c for c in animals if c.endswith("_pct") or c=="Mean_Dark_Structures_per_Macrophage"]
    groups=[]
    if animals.Treatment.eq("").any() or animals.Timepoint.eq("").any():
        return animals,pd.DataFrame()
    for (treatment,timepoint),part in animals.groupby(["Treatment","Timepoint"],sort=False):
        for metric in metrics:
            values=part[metric].dropna()
            groups.append({"Treatment":treatment,"Timepoint":timepoint,"Metric":metric,
                           "N_Animals":len(part),"N_Defined":len(values),"Mean":values.mean(),
                           "SD":values.std(ddof=1)})
    return animals,pd.DataFrame(groups)


class Models:
    def __init__(self, config, resolved_cellpose_path=None, resolved_execution_plan=None):
        import torch
        import timm
        from cellpose.models import CellposeModel
        self.config=config
        self.execution_plan=resolved_execution_plan or execution_plan(config,torch)
        self.device=torch.device(self.execution_plan["cellpose_device"])
        self.classifier_device=torch.device(self.execution_plan["classifier_device"])
        segmentation_weights=resolved_cellpose_path or resolve_cellpose_checkpoint(config.cellpose_checkpoint)
        print("Loading Cellpose-SAM weights...",flush=True)
        self.seg=CellposeModel(pretrained_model=str(segmentation_weights),
                              gpu=self.device.type in ("cuda","mps"),device=self.device,
                              use_bfloat16=self.execution_plan["cellpose_dtype"]=="bfloat16")
        def classifier(path,n):
            print(f"Loading {Path(path).name}...",flush=True)
            model=timm.create_model("vit_base_patch14_dinov2",pretrained=False,num_classes=n,
                                    img_size=224,global_pool="avg")
            state=torch.load(Path(path),map_location="cpu",weights_only=True)
            if not isinstance(state,dict) or "head.weight" not in state:
                raise ValueError(f"Expected the plain trained state_dict in {path}; do not supply only the pretrained backbone.")
            try:
                model.load_state_dict(state,strict=True)
            except RuntimeError as error:
                raise RuntimeError(f"Checkpoint incompatible with the {n}-class DinoBloom-Base model: {path}. "
                                   "Use the original production/fine-tuned weights. No unmatched weights were accepted.") from error
            del state
            return model.float().to(self.classifier_device).eval()
        self.classifier=classifier(config.classifier_checkpoint,8)
        self.expert=classifier(config.expert_checkpoint,3) if config.enable_expert else None
        self.device_name=f"Cellpose={self.device}; classifiers={self.classifier_device}"

    def segment(self, padded):
        result=self.seg.eval(padded,channel_axis=-1,diameter=None,
                             batch_size=self.config.cellpose_batch_size,
                             cellprob_threshold=self.config.cellprob_threshold,
                             flow_threshold=self.config.flow_threshold)
        return result[0] if isinstance(result,tuple) else result

    def predict(self,crops,expert=False):
        import torch
        array=np.stack(crops).astype(np.float32)/255.0
        array=(array-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        batch=torch.from_numpy(np.ascontiguousarray(array.transpose(0,3,1,2))).to(self.classifier_device)
        model=self.expert if expert else self.classifier
        if model is None:
            raise RuntimeError("Expert is disabled.")
        with torch.inference_mode():
            if expert and self.config.expert_tta:
                logits=sum(model(torch.rot90(batch,k,[2,3])) for k in range(4))/4
            else:
                logits=model(batch)
            result=torch.softmax(logits,dim=1).cpu().numpy().astype(np.float32)
        return result


def draw_annotation(canvas, cell, record, width=1, offset=(0,0)):
    import cv2
    ox,oy=offset; color=COLORS[record["Class"]]
    polygons=[np.array(xy,np.int32).reshape(-1,1,2)-np.array([[[ox,oy]]]) for xy in cell["contours_xy"]]
    cv2.drawContours(canvas,polygons,-1,color,width,cv2.LINE_AA)
    x=max(0,min(canvas.shape[1]-1,record["BBox_MinX"]-ox))
    y=max(12,min(canvas.shape[0]-1,record["BBox_MinY"]-oy-3))
    label=annotation_text(record)
    scale=.35; thick=1
    (tw,th),base=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,scale,thick)
    x=max(0,min(x,canvas.shape[1]-tw-2))
    y=max(th+2,min(y,canvas.shape[0]-1-base))
    cv2.rectangle(canvas,(x,max(0,y-th-2)),
                  (min(canvas.shape[1]-1,x+tw+2),min(canvas.shape[0]-1,y+base)),(255,255,255),-1)
    cv2.putText(canvas,label,(x+1,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,thick,cv2.LINE_AA)


def completed_run(folder, fingerprint):
    """Validate all exported bytes before reusing a previous completed run."""
    try:
        report=json.loads((Path(folder)/"RUN_REPORT.json").read_text(encoding="utf-8"))
        if report.get("status")!="completed" or report.get("fingerprint")!=fingerprint:
            return False
        required={"cells.csv","neutrophil_cells.csv","slide_summary.csv","external_contours.json"}
        if report.get("config",{}).get("save_annotated_tiff"):
            required.add("annotated_contours.tif")
        if not required.issubset(report.get("output_hashes",{})):
            return False
        for name,digest in report["output_hashes"].items():
            path=Path(folder)/name
            if not path.is_file() or sha256(path)!=digest:
                return False
        pd.read_csv(Path(folder)/"cells.csv",nrows=1)
        summary=pd.read_csv(Path(folder)/"slide_summary.csv")
        return len(summary)==1
    except (OSError,ValueError,KeyError,pd.errors.ParserError,pd.errors.EmptyDataError):
        return False


class Pipeline:
    def __init__(self,config,models=None):
        config.validate()
        self.config=config
        self.models=models
        self.model_hashes={}
        self.resolved_cellpose_path=None
        if models is None:
            self.execution_plan=execution_plan(config)
            for name,path in [("classifier",config.classifier_checkpoint)]+(
                    [("expert",config.expert_checkpoint)] if config.enable_expert else []):
                if not Path(path).is_file():
                    raise FileNotFoundError(f"Required trained classifier not found: {path}")
                self.model_hashes[name]=sha256(Path(path))
            self.resolved_cellpose_path=resolve_cellpose_checkpoint(config.cellpose_checkpoint)
            self.model_hashes["cellpose"]=sha256(self.resolved_cellpose_path)
        else:
            self.model_hashes={"injected_test_models":type(models).__name__}
            self.execution_plan=getattr(models,"execution_plan",{
                "cellpose_device":models.device_name,"classifier_device":models.device_name,
                "cellpose_dtype":"injected-test-model","classifier_dtype":"injected-test-model"})
        self.source_hash=sha256(__file__)

    def process(self,sample,mode="preview",preview_tiles=4,start_tile=0):
        if mode not in ("preview","full") or preview_tiles<1 or start_tile<0:
            raise ValueError("Use mode preview/full, preview_tiles >=1 and start_tile >=0.")
        info=inspect_slide(sample["path"])
        all_tiles=[(y,min(info["height"],y+self.config.tile_size),x,min(info["width"],x+self.config.tile_size))
                   for y in range(0,info["height"],self.config.tile_size)
                   for x in range(0,info["width"],self.config.tile_size)]
        tiles=all_tiles if mode=="full" else all_tiles[start_tile:start_tile+preview_tiles]
        if not tiles:
            raise ValueError(f"Preview start tile is outside this slide ({len(all_tiles)} tiles total).")
        # Preview is intentionally never promoted into full-slide biological totals.
        scope="full" if mode=="full" else "partial"
        settings={"version":VERSION,"source_sha256":self.source_hash,"config":dataclasses.asdict(self.config),
                  "execution_plan":self.execution_plan,
                  "cellpose_weights_resolved":str(self.resolved_cellpose_path) if self.resolved_cellpose_path else None,
                  "sample":sample,"mode":mode,"preview_tiles":preview_tiles if mode=="preview" else None,
                  "start_tile":start_tile if mode=="preview" else None,"model_hashes":self.model_hashes,
                  "input_sha256":sha256(sample["path"]),"software":versions()}
        fingerprint=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
        parent=Path(self.config.output_dir).expanduser().resolve()/mode
        parent.mkdir(parents=True,exist_ok=True)
        prefix=f"{safe_name(sample['path'])}__{fingerprint[:12]}"
        for candidate in sorted(parent.glob(prefix+"*")):
            if completed_run(candidate,fingerprint):
                print(f"Reusing verified completed {mode} run: {candidate.name}",flush=True)
                return {"folder":str(candidate),"summary":pd.read_csv(candidate/"slide_summary.csv",keep_default_na=False,
                    dtype={"Animal_ID":str,"Treatment":str,"Timepoint":str}).iloc[0].to_dict()}
        folder=parent/f"{prefix}__{uuid.uuid4().hex[:8]}"
        folder.mkdir()
        report={**settings,"fingerprint":fingerprint,"status":"running","scope":scope,
                "started_utc":pd.Timestamp.now(tz="UTC").isoformat(),"image":info,
                "tiles_available":len(all_tiles),"tiles_selected":len(tiles),"tiles_visited":0,
                "input_coordinates":"pixels relative to the decoded RGB image top-left; x right, y down",
                "particle_measurement":"full 224x224 crop dark-structure proxy; not a direct nanotube count",
                "expert_measurement":"image-based reference categories; not direct Siglec-F expression"}
        json_write(folder/"RUN_REPORT.json",report)
        started=time.monotonic()
        try:
            if self.models is None:
                self.models=Models(self.config,resolved_cellpose_path=self.resolved_cellpose_path,
                                   resolved_execution_plan=self.execution_plan)
            report["actual_device"]=self.models.device_name
            with SlideImage(sample["path"],folder/".work") as slide:
                result=self._analyse_image(slide.rgb,tiles,sample,folder,report)
            report.update(result)
            report["elapsed_minutes"]=(time.monotonic()-started)/60
            outputs=[p for p in folder.rglob("*") if p.is_file() and p.name!="RUN_REPORT.json" and ".work" not in p.parts]
            report["output_hashes"]={str(p.relative_to(folder)):sha256(p) for p in outputs}
            report["status"]="completed"
            report["finished_utc"]=pd.Timestamp.now(tz="UTC").isoformat()
            json_write(folder/"RUN_REPORT.json",report)
            print(f"Completed {mode}: {report['retained_cells']:,} cells; {folder}",flush=True)
            return {"folder":str(folder),"summary":pd.read_csv(folder/"slide_summary.csv",keep_default_na=False,
                dtype={"Animal_ID":str,"Treatment":str,"Timepoint":str}).iloc[0].to_dict()}
        except BaseException as error:
            report.update(status="interrupted" if isinstance(error,KeyboardInterrupt) else "failed",
                          error=str(error),traceback=traceback.format_exc())
            json_write(folder/"RUN_REPORT.json",report)
            raise

    def _analyse_image(self,image,tiles,sample,folder,report):
        import cv2
        import tifffile
        from PIL import Image
        cfg=self.config
        cells=[]; skipped=0
        for i,(y1,y2,x1,x2) in enumerate(tiles,1):
            chunk=np.asarray(image[y1:y2,x1:x2])
            mean=float(chunk.mean())
            if mean>245 or mean<10:
                skipped+=1
            else:
                padded=cv2.copyMakeBorder(chunk,cfg.padding,cfg.padding,cfg.padding,cfg.padding,cv2.BORDER_REFLECT)
                masks=self.models.segment(padded)
                masks=np.asarray(masks)[cfg.padding:cfg.padding+chunk.shape[0],cfg.padding:cfg.padding+chunk.shape[1]]
                if masks.shape != chunk.shape[:2]:
                    raise RuntimeError("Cellpose returned an unexpected mask shape.")
                cells.extend(cells_from_masks(masks,y1,x1,cfg.min_area,len(cells)))
            report["tiles_visited"]=i
            json_write(folder/"RUN_REPORT.json",report)
            print(f"Segmentation {i}/{len(tiles)} tiles | {len(cells):,} candidates | {skipped} background tiles",flush=True)
        raw_count=len(cells)
        cells=deduplicate(cells,cfg.nms_iou,cfg.tile_size)
        unique_count=len(cells)
        records=[]; kept_cells=[]; crop_batch=[]; cell_batch=[]
        def flush():
            if not crop_batch:
                return
            probabilities=self.models.predict(crop_batch)
            if np.shape(probabilities)!=(len(crop_batch),len(CLASSES)):
                raise RuntimeError("Classifier output has incorrect dimensions.")
            start=len(records)
            batch_records=[]
            for j,(cell,crop,probs) in enumerate(zip(cell_batch,crop_batch,probabilities)):
                record,_=make_record(cell,probs,crop,cfg,sample,start+j+1)
                batch_records.append(record)
            indices=[i for i,r in enumerate(batch_records) if r["Class"]=="Neutrophil"]
            if indices and cfg.enable_expert:
                expert_probs=self.models.predict([crop_batch[i] for i in indices],expert=True)
                if np.shape(expert_probs)!=(len(indices),3):
                    raise RuntimeError("Expert output has incorrect dimensions.")
                for i,probs in zip(indices,expert_probs):
                    if not np.isfinite(probs).all() or not np.isclose(np.sum(probs),1,atol=1e-4) or np.min(probs)<0:
                        raise RuntimeError("Expert returned invalid probabilities.")
                    idx=int(np.argmax(probs)); r=batch_records[i]
                    r["Expert_Class"]=EXPERT_CLASSES[idx]; r["Expert_Confidence"]=float(probs[idx])
                    r["Expert_Entropy"]=float(-np.sum(np.clip(probs,1e-12,1)*np.log(np.clip(probs,1e-12,1))))
                    r.update({f"Expert_Prob_{c}":float(p) for c,p in zip(EXPERT_CLASSES,probs)})
            records.extend(batch_records);kept_cells.extend(cell_batch)
            print(f"Classification {len(records):,} retained cells",flush=True)
            crop_batch.clear();cell_batch.clear()
        for cell in cells:
            crop=crop_at(image,cell["cy"],cell["cx"])
            if valid_crop(crop,cfg):
                crop_batch.append(crop);cell_batch.append(cell)
            if len(crop_batch)>=cfg.batch_size:
                flush()
        flush()
        df=pd.DataFrame(records,columns=CELL_COLUMNS)
        df.to_csv(folder/"cells.csv",index=False)
        df[df.Class=="Neutrophil"].to_csv(folder/"neutrophil_cells.csv",index=False)
        summary=summarise_cells(df,sample,cfg.enable_expert)
        summary.update(Scope=report["scope"],Tiles_Visited=report["tiles_visited"],Tiles_Available=report["tiles_available"])
        pd.DataFrame([summary]).to_csv(folder/"slide_summary.csv",index=False)
        json_write(folder/"external_contours.json",{
            "coordinates":"global image pixels, x then y; external polygons only, holes not retained",
            "scope":report["scope"],"cells":[{"Cell_ID":r["Cell_ID"],"Class":r["Class"],
               "Model_Predicted_Class":r["Model_Predicted_Class"],"Model_Confidence":r["Model_Confidence"],
               "contours_xy":c["contours_xy"]} for c,r in zip(kept_cells,records)]})
        if cfg.save_preview_pairs:
            preview_dir=folder/"previews";preview_dir.mkdir()
            selected=[]
            for tile in tiles:
                y1,y2,x1,x2=tile
                items=[(c,r) for c,r in zip(kept_cells,records) if y1<=c["cy"]<y2 and x1<=c["cx"]<x2]
                if items:
                    selected.append((tile,items))
                if len(selected)>=cfg.save_preview_pairs:
                    break
            if not selected:
                selected=[(tiles[0],[])]
            for i,((y1,y2,x1,x2),items) in enumerate(selected,1):
                original=np.array(image[y1:y2,x1:x2],copy=True);annotated=original.copy()
                for c,r in items:
                    draw_annotation(annotated,c,r,cfg.contour_width,(x1,y1))
                Image.fromarray(original).save(preview_dir/f"{i:02d}_original.png")
                Image.fromarray(annotated).save(preview_dir/f"{i:02d}_annotated.png")
        if cfg.save_annotated_tiff:
            scratch=folder/".work"/"annotated.dat"
            canvas=np.memmap(scratch,mode="w+",dtype=np.uint8,shape=image.shape)
            try:
                for y in range(0,image.shape[0],512):
                    canvas[y:y+512]=image[y:y+512]
                for c,r in zip(kept_cells,records):
                    draw_annotation(canvas,c,r,cfg.contour_width)
                def tile_iterator():
                    for y in range(0,image.shape[0],256):
                        for x in range(0,image.shape[1],256):
                            tile=np.zeros((256,256,3),np.uint8)
                            sub=canvas[y:y+256,x:x+256]
                            tile[:sub.shape[0],:sub.shape[1]]=sub
                            yield tile
                physical={}
                spacing=report["image"].get("pixel_size_um",{})
                if all(k in spacing and np.isfinite(spacing[k]) and spacing[k]>0 for k in ("X","Y")):
                    physical={"resolution":(10000/spacing["X"],10000/spacing["Y"]),"resolutionunit":"CENTIMETER"}
                tifffile.imwrite(folder/"annotated_contours.tif",data=tile_iterator(),shape=image.shape,
                                 dtype=np.uint8,tile=(256,256),photometric="rgb",bigtiff=True,
                                 compression="deflate",metadata={"axes":"YXS","scope":report["scope"],
                                 "pixel_size_um":spacing,
                                 "note":"Class-coloured contours and labels; neutrophil subtypes when enabled; P is the detected dark-structure count. Probabilities are in cells.csv. No multiresolution pyramid."},**physical)
            finally:
                canvas.flush();canvas._mmap.close();del canvas
                scratch.unlink(missing_ok=True)
        return {"raw_candidates":raw_count,"unique_candidates":unique_count,"retained_cells":len(df),
                "filtered_cells":unique_count-len(df),"background_tiles":skipped,
                "expert_status":"disabled" if not cfg.enable_expert else (
                    "no_neutrophils" if summary["Count_Neutrophil"]==0 else "completed"),
                "annotation_tiff":"full-resolution tiled BigTIFF, no pyramid" if cfg.save_annotated_tiff else "disabled"}

    def run(self,samples,mode="preview",preview_tiles=4,start_tile=0):
        # Aggregate only this explicit invocation, never every old file in a folder.
        if not samples:
            raise ValueError("No input samples selected.")
        results=[]
        selected=samples[:1] if mode=="preview" else samples
        for sample in selected:
            results.append(self.process(sample,mode,preview_tiles,start_tile))
            gc.collect()
        rows=[r["summary"] for r in results]
        slides=pd.DataFrame(rows)
        # Read actual summaries to preserve NaN and boolean dtypes for aggregation.
        slides=pd.concat([pd.read_csv(Path(r["folder"])/"slide_summary.csv",keep_default_na=False,
                                     na_values=[""],dtype={"Animal_ID":str,"Treatment":str,"Timepoint":str})
                          for r in results],ignore_index=True)
        for col in ("Animal_ID","Treatment","Timepoint"):
            slides[col]=slides[col].fillna("").astype(str)
        batch=Path(self.config.output_dir)/"batches"/(pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:6])
        batch.mkdir(parents=True)
        slides.to_csv(batch/("ALL_SLIDE_STATISTICS.csv" if mode=="full" else "PARTIAL_SLIDE_STATISTICS.csv"),index=False)
        if self.config.save_annotated_tiff:
            pd.DataFrame([{"Slide":r["summary"]["Slide"],"Scope":r["summary"]["Scope"],
                           "Annotated_TIFF":str((Path(r["folder"])/"annotated_contours.tif").resolve())}
                          for r in results]).to_csv(batch/"TIFF_OUTPUTS.csv",index=False)
        if mode=="full":
            animals,groups=aggregate_animals(slides)
            if not animals.empty:
                animals.to_csv(batch/"ANIMAL_STATISTICS.csv",index=False)
            if not groups.empty:
                groups.to_csv(batch/"GROUP_MEAN_SD.csv",index=False,float_format="%.8f")
            else:
                print("Group summaries omitted: complete animal/treatment/time metadata are required.")
        json_write(batch/"BATCH_REPORT.json",{"version":VERSION,"mode":mode,"status":"completed",
            "selected_inputs":[s["path"] for s in selected],"completed_runs":[r["folder"] for r in results],
            "animal_aggregation":"Counts pooled within explicitly identified animals; group mean and sample SD across animals."})
        print(f"Batch results: {batch}",flush=True)
        return {"batch_folder":str(batch),"runs":results,"slide_statistics":slides}
