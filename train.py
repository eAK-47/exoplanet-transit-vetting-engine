#!/usr/bin/env python3
"""
INS Vikramadithya — Standalone High-Accuracy Training Pipeline
==============================================================

Trains the SwinTransitClassifier from scratch on a synthetic dataset of
phase-folded light curve morphologies using mixed-precision (FP16) CUDA
acceleration on the local RTX 4050 Tensor Core GPU.

Workflow:
  1. Generate 1,000 synthetic 2D transit images (500 Planet / 500 EB False Positive)
  2. Build PyTorch Dataset + DataLoader with ImageNet normalization
  3. Train SwinTransitClassifier for 8 epochs with AdamW + CrossEntropyLoss
  4. Export trained weights to ``model_weights.pth``
  5. Patch ``src/models/inference.py`` to auto-load weights on init

Author: INS Vikramadithya Pipeline Team
"""

import os
import sys
import math
import logging
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import autocast, GradScaler

# ---------------------------------------------------------------------------
# Add project root to sys.path so we can import src.models.inference
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.inference import SwinTransitClassifier

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
NUM_CLASSES = 2                     # 0 = Planet, 1 = False Positive (EB)
IMG_SIZE = 224                      # Swin-Tiny expects 224x224
SAMPLES_PER_CLASS = 500             # 500 each → 1,000 total
TRAIN_SPLIT = 0.8                   # 80 % train, 20 % validation
BATCH_SIZE = 64                     # Max for RTX 4050 6 GB with FP16
NUM_EPOCHS = 8
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4

SYNTHETIC_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "model_weights.pth")

# ImageNet normalisation statistics (used by Swin Transformer)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===================================================================
#  1.  DATA SYNTHESIS ENGINE
# ===================================================================

def _gaussian_noise(size: Tuple[int, ...], sigma: float = 0.003) -> np.ndarray:
    """Additive white Gaussian noise."""
    return np.random.normal(loc=0.0, scale=sigma, size=size).astype(np.float32)


def _trapezoid_transit(
    phase: np.ndarray,
    depth: float,
    ingress_egress_frac: float = 0.15,
) -> np.ndarray:
    """
    Generate a flat-bottomed U-shape (trapezoidal) transit profile.

    Parameters
    ----------
    phase : np.ndarray
        Normalised phase array in [-0.5, 0.5].
    depth : float
        Fractional depth of the transit (e.g. 0.01 for 1 %).
    ingress_egress_frac : float
        Fraction of the total transit duration spent in ingress/egress.

    Returns
    -------
    flux : np.ndarray
        Flux values with the transit profile applied (baseline = 1.0).
    """
    # Transit width in phase units (randomised per call)
    half_width = np.random.uniform(0.04, 0.10)
    ingress_half = half_width * ingress_egress_frac

    flux = np.ones_like(phase, dtype=np.float32)

    # Ingress region
    mask_ingress = (phase >= -half_width) & (phase < -half_width + ingress_half)
    flux[mask_ingress] = 1.0 - depth * (phase[mask_ingress] + half_width) / ingress_half

    # Flat bottom
    mask_bottom = (phase >= -half_width + ingress_half) & (phase <= half_width - ingress_half)
    flux[mask_bottom] = 1.0 - depth

    # Egress region
    mask_egress = (phase > half_width - ingress_half) & (phase <= half_width)
    flux[mask_egress] = 1.0 - depth * (half_width - phase[mask_egress]) / ingress_half

    return flux


def _vshape_transit(phase: np.ndarray, depth: float) -> np.ndarray:
    """
    Generate a sharp V-shape (triangular) eclipse profile typical of
    eclipsing binaries / false positives.

    Parameters
    ----------
    phase : np.ndarray
        Normalised phase array in [-0.5, 0.5].
    depth : float
        Fractional depth at the centre of the eclipse.

    Returns
    -------
    flux : np.ndarray
        Flux values with the V-shape profile applied (baseline = 1.0).
    """
    half_width = np.random.uniform(0.02, 0.06)
    flux = np.ones_like(phase, dtype=np.float32)

    mask = np.abs(phase) <= half_width
    # Linear V: depth scales linearly from 0 at edges to full depth at centre
    flux[mask] = 1.0 - depth * (1.0 - np.abs(phase[mask]) / half_width)

    return flux


def _render_as_image(phase: np.ndarray, flux: np.ndarray) -> np.ndarray:
    """
    Render a 1-D phase-folded light curve as a 224x224 RGB image array.

    The image is a scatter-plot style visualisation on a white background
    with black points, normalised to [0, 1] and then converted to 3-channel
    RGB with ImageNet-style normalisation applied later by the Dataset.

    Returns
    -------
    image : np.ndarray, shape (3, 224, 224), dtype=np.float32, range [0, 1]
    """
    # Create a white canvas
    canvas = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    # Map phase [-0.5, 0.5] → pixel columns [5, 219]
    x = ((phase + 0.5) * (IMG_SIZE - 10) + 5).astype(np.int32)
    x = np.clip(x, 0, IMG_SIZE - 1)

    # Map flux to pixel rows: lower flux → higher row (inverted y-axis)
    # Flux range: typical dips are 0.95–1.0, so map [0.94, 1.01] → [5, 218]
    y_min, y_max = 0.94, 1.01
    y = IMG_SIZE - 5 - ((flux - y_min) / (y_max - y_min) * (IMG_SIZE - 10)).astype(np.int32)
    y = np.clip(y, 0, IMG_SIZE - 1)

    # Draw points (3x3 pixel blobs for visibility)
    for xi, yi in zip(x, y):
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                px, py = xi + dx, yi + dy
                if 0 <= px < IMG_SIZE and 0 <= py < IMG_SIZE:
                    canvas[py, px] = 0.0  # black point

    # Stack into 3-channel RGB (grayscale)
    image = np.stack([canvas, canvas, canvas], axis=0).astype(np.float32)  # (3, 224, 224)
    return image


def generate_synthetic_dataset(
    save_dir: str = SYNTHETIC_DATA_DIR,
    samples_per_class: int = SAMPLES_PER_CLASS,
    num_points: int = 500,
) -> None:
    """
    Generate and save a synthetic dataset of phase-folded transit images.

    Class 0 — Planet transit:  trapezoidal U-shape dip
    Class 1 — EB false positive:  triangular V-shape dip

    Each sample is saved as a ``.npy`` file with shape ``(3, 224, 224)``.
    A companion CSV label file is also written for easy loading.
    """
    os.makedirs(save_dir, exist_ok=True)
    logger.info("=" * 60)
    logger.info("DATA SYNTHESIS ENGINE")
    logger.info(f"Generating {2 * samples_per_class} synthetic transit images...")
    logger.info(f"  Class 0: Planet transit  (U-shape, {samples_per_class} samples)")
    logger.info(f"  Class 1: EB false positive (V-shape, {samples_per_class} samples)")
    logger.info(f"  Saving to: {save_dir}")
    logger.info("=" * 60)

    labels_csv = []

    for class_id, class_name in [(0, "planet"), (1, "eb_false_positive")]:
        for i in range(samples_per_class):
            # Random phase array
            phase = np.linspace(-0.5, 0.5, num_points, dtype=np.float32)

            # Random depth between 0.5 % and 4 %
            depth = np.random.uniform(0.005, 0.04)

            if class_id == 0:
                # Planet: trapezoidal U-shape
                flux = _trapezoid_transit(phase, depth)
            else:
                # EB: triangular V-shape
                flux = _vshape_transit(phase, depth)

            # Add Gaussian white noise
            noise_sigma = np.random.uniform(0.001, 0.005)
            flux += _gaussian_noise(flux.shape, sigma=noise_sigma)

            # Render as 224x224 RGB image
            image = _render_as_image(phase, flux)  # (3, 224, 224), range [0, 1]

            # Save
            filename = f"{class_name}_{i:04d}.npy"
            filepath = os.path.join(save_dir, filename)
            np.save(filepath, image)

            labels_csv.append(f"{filename},{class_id}")

            if (i + 1) % 100 == 0:
                logger.info(f"  [{class_name}] {i + 1:4d} / {samples_per_class} generated")

    # Write label file
    label_path = os.path.join(save_dir, "labels.csv")
    with open(label_path, "w") as f:
        f.write("filename,label\n")
        for line in labels_csv:
            f.write(line + "\n")

    logger.info(f"Label file written to: {label_path}")
    logger.info(f"Total samples generated: {2 * samples_per_class}")
    logger.info("Data synthesis complete.\n")


# ===================================================================
#  2.  PYTORCH DATASET
# ===================================================================

class SyntheticTransitDataset(Dataset):
    """
    PyTorch Dataset that loads pre-generated synthetic transit images
    from ``.npy`` files, applies ImageNet normalisation, and returns
    tensors of shape ``(3, 224, 224)``.
    """

    def __init__(self, data_dir: str = SYNTHETIC_DATA_DIR, split: str = "train"):
        """
        Parameters
        ----------
        data_dir : str
            Directory containing ``.npy`` image files and ``labels.csv``.
        split : str
            One of ``"train"`` or ``"val"``.
        """
        self.data_dir = data_dir
        self.split = split

        # Load label file
        label_path = os.path.join(data_dir, "labels.csv")
        if not os.path.exists(label_path):
            raise FileNotFoundError(
                f"Labels file not found at {label_path}. "
                "Run generate_synthetic_dataset() first."
            )

        with open(label_path, "r") as f:
            lines = f.readlines()[1:]  # skip header

        all_files: List[str] = []
        all_labels: List[int] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            fname, lbl = line.split(",")
            all_files.append(fname)
            all_labels.append(int(lbl))

        # Train / val split (80/20 stratified)
        rng = np.random.RandomState(SEED)
        indices = np.arange(len(all_files))
        rng.shuffle(indices)

        split_idx = int(TRAIN_SPLIT * len(indices))
        if split == "train":
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]

        self.files = [all_files[i] for i in self.indices]
        self.labels = [all_labels[i] for i in self.indices]

        logger.info(
            f"SyntheticTransitDataset [{split}]: {len(self.files)} samples loaded."
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image (3, 224, 224) float32 in [0, 1]
        img_path = os.path.join(self.data_dir, self.files[idx])
        image = np.load(img_path).astype(np.float32)  # (3, 224, 224)

        # ImageNet normalisation
        for c in range(3):
            image[c] = (image[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

        label = self.labels[idx]

        return torch.from_numpy(image), torch.tensor(label, dtype=torch.long)


# ===================================================================
#  3.  TRAINING LOOP
# ===================================================================

def train() -> None:
    """
    Main training routine.

    1. Generates (or loads cached) synthetic dataset
    2. Builds DataLoaders
    3. Initialises SwinTransitClassifier, AdamW, CrossEntropyLoss
    4. Runs mixed-precision (FP16) training loop for 8 epochs
    5. Saves ``model_weights.pth``
    """
    # ------------------------------------------------------------------
    # Device detection
    # ------------------------------------------------------------------
    if not torch.cuda.is_available():
        logger.error(
            "CUDA is NOT available. This pipeline requires an NVIDIA GPU "
            "with CUDA support for mixed-precision training. "
            "Install the CUDA-enabled PyTorch build and verify with:\n"
            "    python -c \"import torch; print(torch.cuda.is_available())\""
        )
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"Using GPU: {gpu_name}")
    logger.info(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    logger.info(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # ------------------------------------------------------------------
    # 3a. Generate / verify synthetic dataset
    # ------------------------------------------------------------------
    if not os.path.exists(os.path.join(SYNTHETIC_DATA_DIR, "labels.csv")):
        logger.info("Synthetic dataset not found. Generating now...")
        generate_synthetic_dataset()
    else:
        logger.info(f"Using existing synthetic dataset at {SYNTHETIC_DATA_DIR}")

    # ------------------------------------------------------------------
    # 3b. Datasets & DataLoaders
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("DATALOADER SETUP")
    logger.info("=" * 60)

    train_dataset = SyntheticTransitDataset(SYNTHETIC_DATA_DIR, split="train")
    val_dataset = SyntheticTransitDataset(SYNTHETIC_DATA_DIR, split="val")

    # Determine maximum batch size that fits in VRAM
    # RTX 4050 has 6 GB — Swin-Tiny at FP16 with batch 64 fits comfortably
    batch_size = BATCH_SIZE
    logger.info(f"Batch size: {batch_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,          # 0 for Windows stability
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    logger.info(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")
    logger.info(f"Train samples: {len(train_dataset)}  |  Val samples: {len(val_dataset)}")

    # ------------------------------------------------------------------
    # 3c. Model, Optimizer, Loss, Scaler
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("MODEL INITIALISATION")
    logger.info("=" * 60)

    model = SwinTransitClassifier()
    model = model.to(device)

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda")

    # ------------------------------------------------------------------
    # 3d. Training Loop
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("TRAINING LOOP — Mixed Precision (FP16) on CUDA")
    logger.info(f"Epochs: {NUM_EPOCHS}  |  Optimizer: AdamW  |  LR: {LEARNING_RATE}")
    logger.info("=" * 60)

    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Mixed-precision forward pass
            with autocast(device_type="cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            num_batches += 1

            # Log every 5 batches
            if (batch_idx + 1) % 5 == 0:
                logger.info(
                    f"  Epoch {epoch:2d}/{NUM_EPOCHS}  |  Batch {batch_idx + 1:3d}/{len(train_loader)}  "
                    f"|  Loss: {loss.item():.4f}"
                )

        avg_train_loss = running_loss / num_batches

        # --- Validation Phase ---
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast(device_type="cuda"):
                    logits = model(images)
                    preds = torch.argmax(logits, dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = 100.0 * correct / total

        # --- Epoch Summary ---
        logger.info("-" * 60)
        logger.info(
            f"Epoch {epoch:2d}/{NUM_EPOCHS}  |  "
            f"Train Loss: {avg_train_loss:.6f}  |  "
            f"Val Accuracy: {correct}/{total} = {val_acc:.2f}%"
        )
        logger.info("-" * 60)

        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logger.info(f"  ★ New best validation accuracy: {best_val_acc:.2f}%")

    # ------------------------------------------------------------------
    # 4.  WEIGHT EXPORT
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("WEIGHT EXPORT")
    logger.info("=" * 60)

    # Save full model state dict
    torch.save(model.state_dict(), WEIGHTS_PATH)
    logger.info(f"Model weights saved to: {WEIGHTS_PATH}")
    logger.info(f"  File size: {os.path.getsize(WEIGHTS_PATH) / 1e6:.2f} MB")

    # Also save a CPU-compatible copy for inference on CPU
    cpu_weights_path = os.path.join(PROJECT_ROOT, "model_weights_cpu.pth")
    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(cpu_state, cpu_weights_path)
    logger.info(f"CPU-compatible weights also saved to: {cpu_weights_path}")

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 5.  INFERENCE PATCH INSTRUCTION
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("INFERENCE PATCH INSTRUCTION")
    logger.info("=" * 60)
    logger.info(
        "To make SwinTransitClassifier auto-load trained weights, add the "
        "following logic to the __init__ method of SwinTransitClassifier "
        "in src/models/inference.py:\n"
    )
    logger.info(
        "    import os\n"
        "    weights_path = os.path.join(os.path.dirname(__file__), "
        "'..', '..', 'model_weights.pth')\n"
        "    if os.path.exists(weights_path):\n"
        "        state = torch.load(weights_path, map_location='cpu')\n"
        "        self.load_state_dict(state)\n"
        "        logger.info(f'Loaded trained weights from {weights_path}')\n"
    )
    logger.info("=" * 60)


# ===================================================================
#  ENTRY POINT
# ===================================================================

if __name__ == "__main__":
    train()