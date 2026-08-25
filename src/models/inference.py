from __future__ import annotations

# ---------------------------------------------------------------------------
# Lazy imports for heavy ML dependencies (torch, transformers).
# The module must be importable in CI without these installed — tests and the
# API server import this module at module level, so we cannot hard-import at
# top scope. Instead we set availability flags and raise clear errors when
# ML functionality is requested without the dependencies.
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None           # type: ignore[assignment]
    nn = None              # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    from transformers import SwinModel, AutoImageProcessor
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    SwinModel = None       # type: ignore[assignment]
    AutoImageProcessor = None  # type: ignore[assignment]
    _TRANSFORMERS_AVAILABLE = False

# Non-ML dependencies are cheap enough to import eagerly.
import os
import logging
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("inference")

# ---------------------------------------------------------------------------
# NASA Exoplanet Archive PS CompPars table URL (TAP service)
# ---------------------------------------------------------------------------
_EXOPLANET_ARCHIVE_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
    "query=select+pl_name,pl_orbper,pl_trandep,st_teff+from+pscomppars&format=csv"
)
_CATALOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "exoplanet_catalog.csv"
)
_CATALOG_PATH = os.path.normpath(_CATALOG_PATH)

# ---------------------------------------------------------------------------
# Fallback dataset — 20 real exoplanets spanning a wide parameter range
# Used when the NASA Archive network query fails and no cached CSV exists.
# Columns: name, period (days), depth (PPM), teff (K), type, discovery_year
# ---------------------------------------------------------------------------
_FALLBACK_CATALOG = [
    # Ultra-short-period planets (P < 1 day) — depth IN PPM
    {"name": "LTT 9779 b",       "period": 0.792,  "depth": 7500,   "teff": 5443, "type": "Ultra-Hot Neptune", "discovery_year": 2020},
    {"name": "CoRoT-7b",         "period": 0.854,  "depth": 3500,   "teff": 5275, "type": "Super-Earth",       "discovery_year": 2009},
    {"name": "Kepler-78b",       "period": 0.355,  "depth": 2000,   "teff": 5089, "type": "Lava World",        "discovery_year": 2013},
    {"name": "WASP-12b",         "period": 1.091,  "depth": 150000, "teff": 6300, "type": "Hot Jupiter",       "discovery_year": 2008},
    {"name": "55 Cnc e",         "period": 0.737,  "depth": 4000,   "teff": 5196, "type": "Super-Earth",       "discovery_year": 2004},
    # Short-period planets (1-10 days)
    {"name": "HD 209458 b",      "period": 3.525,  "depth": 150000, "teff": 6092, "type": "Hot Jupiter",       "discovery_year": 1999},
    {"name": "HD 189733 b",      "period": 2.219,  "depth": 250000, "teff": 5040, "type": "Hot Jupiter",       "discovery_year": 2005},
    {"name": "WASP-39 b",        "period": 4.055,  "depth": 70000,  "teff": 5400, "type": "Hot Saturn",        "discovery_year": 2011},
    {"name": "TRAPPIST-1e",      "period": 6.100,  "depth": 60000,  "teff": 2559, "type": "Earth-sized (Habitable)", "discovery_year": 2017},
    {"name": "GJ 1214 b",        "period": 1.580,  "depth": 130000, "teff": 3026, "type": "Mini-Neptune",      "discovery_year": 2009},
    # Medium-period planets (10-100 days)
    {"name": "Kepler-186f",      "period": 129.94, "depth": 4000,   "teff": 3755, "type": "Earth-sized",       "discovery_year": 2014},
    {"name": "Kepler-22b",       "period": 289.86, "depth": 5000,   "teff": 5618, "type": "Super-Earth",       "discovery_year": 2011},
    {"name": "Kepler-62f",       "period": 267.29, "depth": 5000,   "teff": 4925, "type": "Super-Earth",       "discovery_year": 2013},
    {"name": "Kepler-442b",      "period": 112.31, "depth": 7000,   "teff": 4402, "type": "Super-Earth",       "discovery_year": 2015},
    {"name": "Kepler-452b",      "period": 384.84, "depth": 4000,   "teff": 5757, "type": "Super-Earth",       "discovery_year": 2015},
    # Long-period / cool planets
    {"name": "Proxima Cen b",    "period": 11.186, "depth": 50000,  "teff": 3042, "type": "Earth-sized",       "discovery_year": 2016},
    {"name": "K2-18 b",          "period": 32.940, "depth": 60000,  "teff": 3457, "type": "Sub-Neptune",       "discovery_year": 2015},
    {"name": "HD 219134 b",      "period": 3.093,  "depth": 4000,   "teff": 4699, "type": "Super-Earth",       "discovery_year": 2015},
    {"name": "TOI-700 d",        "period": 37.426, "depth": 50000,  "teff": 3480, "type": "Earth-sized",       "discovery_year": 2020},
    {"name": "Kepler-10b",       "period": 0.838,  "depth": 2000,   "teff": 5708, "type": "Lava World",        "discovery_year": 2011},
]


# Base class for SwinTransitClassifier — uses nn.Module when torch is available,
# or object as a fallback so the module can be imported without torch.
_SwinTransitClassifierBase = nn.Module if _TORCH_AVAILABLE else object

class SwinTransitClassifier(_SwinTransitClassifierBase):
    """
    Swin Transformer backbone classifier for 2D transit morphology analysis.
    Outputs binary probability: Planet vs. Astrophysical False Positive.
    """
    def __init__(self, pretrained_model_name: str = 'microsoft/swin-tiny-patch4-window7-224'):
        super().__init__()
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required to use SwinTransitClassifier. "
                "Install it with: pip install torch torchvision"
            )
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required to use SwinTransitClassifier. "
                "Install it with: pip install transformers"
            )
        logger.info(f"Initializing Swin Transformer backbone: {pretrained_model_name}")
        self.backbone = SwinModel.from_pretrained(pretrained_model_name)
        hidden_size = self.backbone.config.hidden_size
        # Two classes: [Planet Confidence, False Positive Confidence]
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )

        # Auto-load trained weights if model_weights.pth exists
        weights_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "model_weights.pth"
        )
        weights_path = os.path.normpath(weights_path)
        if os.path.exists(weights_path):
            try:
                state = torch.load(weights_path, map_location="cpu", weights_only=True)
                self.load_state_dict(state)
                logger.info(f"✓ Loaded trained weights from {weights_path}")
            except Exception as e:
                logger.warning(f"Could not load weights from {weights_path}: {e}")
        else:
            logger.info("No trained weights found at %s — using randomly initialized classifier head.", weights_path)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        # Swin model outputs a tuple; pooler_output is the representation of the class token equivalent (shape: batch_size, hidden_size)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits


def _load_exoplanet_catalog() -> Optional[pd.DataFrame]:
    """
    Attempt to load the NASA Exoplanet Archive PS CompPars table.

    Strategy:
      1. If a cached CSV exists at ``data/exoplanet_catalog.csv``, read it.
      2. Otherwise, attempt to download from the public TAP service.
      3. If both fail, return ``None`` (caller falls back to hardcoded array).

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns ``pl_name``, ``pl_orbper``, ``pl_trandep``, ``st_teff``,
        with NaN rows removed.  ``None`` if loading failed entirely.

    Notes
    -----
    The NASA Archive stores ``pl_trandep`` in **percent** units (0-100%).
    We convert to ppm (parts-per-million) by multiplying by 10,000 so that
    the values are consistent with the pipeline's depth estimates.
    """
    # --- Attempt 1: cached CSV ---
    if os.path.exists(_CATALOG_PATH):
        try:
            df = pd.read_csv(_CATALOG_PATH)
            required = {"pl_name", "pl_orbper", "pl_trandep", "st_teff"}
            if required.issubset(df.columns):
                df = df.dropna(subset=list(required)).reset_index(drop=True)
                logger.info(f"Loaded {len(df)} exoplanets from cached CSV: {_CATALOG_PATH}")
                return df
            else:
                logger.warning("Cached CSV missing required columns; re-downloading.")
        except Exception as e:
            logger.warning(f"Failed to read cached CSV: {e}")

    # --- Attempt 2: download from NASA Exoplanet Archive ---
    try:
        logger.info("Downloading exoplanet catalog from NASA Exoplanet Archive TAP service...")
        df = pd.read_csv(_EXOPLANET_ARCHIVE_URL)
        required = {"pl_name", "pl_orbper", "pl_trandep", "st_teff"}
        if required.issubset(df.columns):
            df = df.dropna(subset=list(required)).reset_index(drop=True)
            # Cache to disk
            os.makedirs(os.path.dirname(_CATALOG_PATH), exist_ok=True)
            df.to_csv(_CATALOG_PATH, index=False)
            logger.info(f"Downloaded and cached {len(df)} exoplanets to {_CATALOG_PATH}")
            return df
        else:
            logger.warning(f"Downloaded CSV missing columns. Found: {list(df.columns)}")
    except Exception as e:
        logger.warning(f"NASA Exoplanet Archive download failed: {e}")

    # --- Both attempts failed ---
    logger.info("Falling back to hardcoded exoplanet catalog.")
    return None


class VikramadithyaInferenceEngine:
    """
    Inference Engine wrapper to bind model weights to GPU, run mixed-precision evaluation,
    and match exoplanets against historical records using KNN.
    """
    def __init__(self, model_name: str = 'microsoft/swin-tiny-patch4-window7-224'):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required to use VikramadithyaInferenceEngine. "
                "Install it with: pip install torch torchvision"
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Inference Engine running on device: {self.device}")

        # Initialize the image processor and custom Swin classifier
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(model_name)
            self._has_image_processor = True
            logger.info("Successfully loaded HuggingFace AutoImageProcessor.")
        except Exception as e:
            logger.warning(f"Using manual custom image preprocessing fallback (Torchvision might be absent): {e}")
            self._has_image_processor = False

        self.model = SwinTransitClassifier(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Build 3D KNN Space from the exoplanet catalog
        self._initialize_historical_knn()

    # ------------------------------------------------------------------
    #  NASA Exoplanet Archive → 3D KNN Pipeline
    # ------------------------------------------------------------------

    def _initialize_historical_knn(self):
        """
        Load the exoplanet catalog (NASA Archive or fallback) and build a
        3-dimensional KNN space using [Orbital Period, Transit Depth (ppm),
        Host Star Teff] with RobustScaler normalisation.

        The resulting ``self.historical_db`` is a list of dicts with keys:
            name, period, depth, teff, type, discovery_year

        Notes
        -----
        - NASA Archive stores ``pl_trandep`` in **percent**; we convert to ppm
          (×10,000) for consistency with the pipeline.
        - ``RobustScaler`` (median/IQR) is used instead of ``MinMaxScaler``
          because the data contains extreme outliers (period 0.18–3650d,
          depth 0–566,500ppm, Teff 2566–10170K) that would compress the
          common range under min-max scaling.
        """
        # --- Load catalog ---
        df = _load_exoplanet_catalog()

        if df is not None and len(df) > 0:
            # Use the real NASA Archive data
            self.historical_db = []
            for _, row in df.iterrows():
                self.historical_db.append({
                    "name": str(row["pl_name"]),
                    "period": float(row["pl_orbper"]),
                    # Convert percent → ppm (×10,000)
                    "depth": float(row["pl_trandep"]) * 10000.0,
                    "teff": float(row["st_teff"]),
                    "type": "Archive",
                    "discovery_year": 0,
                })
            logger.info(f"KNN initialized with {len(self.historical_db)} real exoplanets from NASA Archive.")
        else:
            # Fallback to hardcoded dataset (depth already in ppm)
            self.historical_db = [dict(entry) for entry in _FALLBACK_CATALOG]
            logger.info(f"KNN initialized with {len(self.historical_db)} fallback exoplanets.")

        # --- Build 3D feature array: [period, depth_ppm, teff] ---
        self.knn_features = np.array([
            [p["period"], p["depth"], p["teff"]]
            for p in self.historical_db
        ], dtype=np.float64)

        # --- RobustScaler: uses median/IQR, robust to extreme outliers ---
        # Period spans 0.18–3650d, Depth spans 0–566,500ppm, Teff spans 2566–10170K
        # RobustScaler prevents the extreme values from compressing the common range
        self.scaler = RobustScaler()
        scaled_features = self.scaler.fit_transform(self.knn_features)

        # --- Fit NearestNeighbors (k=3, Euclidean) ---
        self.knn = NearestNeighbors(n_neighbors=3, metric="euclidean")
        self.knn.fit(scaled_features)
        logger.info("3D KNN space [Period (d), Depth (ppm), Teff (K)] fitted with RobustScaler.")

    # ------------------------------------------------------------------
    #  Inference
    # ------------------------------------------------------------------

    def evaluate_image(self, img_path: str) -> Dict[str, float]:
        """
        Processes the generated 2D transit morphology image using
        torch.cuda.amp.autocast() for Mixed Precision (FP16) compute optimization.
        """
        logger.info(f"Evaluating transit morphology image: {img_path}")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Transit image does not exist: {img_path}")

        # Open image and convert to RGB
        with Image.open(img_path) as image:
            image = image.convert("RGB")

            if self._has_image_processor:
                # Preprocess using the HuggingFace AutoImageProcessor
                inputs = self.image_processor(images=image, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)
            else:
                # Manual custom preprocessing fallback
                image_resized = image.resize((224, 224), Image.Resampling.BILINEAR)
                img_np = np.array(image_resized).astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                img_np = (img_np - mean) / std
                img_np = img_np.transpose(2, 0, 1)
                pixel_values = torch.tensor(img_np).unsqueeze(0).to(self.device)

        # Run inference using Mixed Precision (autocast) if CUDA is available
        use_amp = (self.device.type == "cuda")

        with torch.amp.autocast("cuda", enabled=use_amp):
            with torch.no_grad():
                logits = self.model(pixel_values)
                probabilities = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        # Class 0: Planet, Class 1: False Positive
        planet_conf = float(probabilities[0])
        fp_conf = float(probabilities[1])

        logger.info(f"Inference output: Planet Confidence = {planet_conf:.4f}, False Positive = {fp_conf:.4f}")
        return {
            "planet_confidence": planet_conf,
            "false_positive_confidence": fp_conf
        }

    # ------------------------------------------------------------------
    #  3D KNN Similarity Search
    # ------------------------------------------------------------------

    def find_nearest_neighbors(
        self,
        period: float,
        transit_depth_ppm: float,
        teff: float,
    ) -> List[Dict[str, Any]]:
        """
        Query the 3D KNN space with the target's [period, depth, teff] and
        return the 3 most similar exoplanets with a normalised similarity
        percentage (100% = perfect match).

        Parameters
        ----------
        period : float
            Orbital period in days.
        transit_depth_ppm : float
            Transit depth in parts-per-million.
        teff : float
            Host star effective temperature in Kelvin.

        Returns
        -------
        List[Dict[str, Any]]
            Each dict contains: name, period, depth, teff, type,
            discovery_year, similarity_pct.
        """
        # Build target feature vector and scale
        target_feat = np.array([[period, transit_depth_ppm, teff]], dtype=np.float64)
        scaled_target = self.scaler.transform(target_feat)

        # Query KNN
        distances, indices = self.knn.kneighbors(scaled_target)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            match = dict(self.historical_db[idx])  # shallow copy
            # Convert Euclidean distance to similarity percentage
            # distance=0 → 100%, distance=1 → 50%, distance=3 → 25%
            similarity_pct = 100.0 / (1.0 + float(dist))
            match["similarity_pct"] = round(similarity_pct, 2)
            results.append(match)

        names_str = ", ".join([f"{m['name']} ({m['similarity_pct']}%)" for m in results])
        logger.info(f"KNN matched top 3: {names_str}")
        return results


if __name__ == "__main__":
    # Small test suite to verify module correctness
    print("Testing Swin Transformer and 3D KNN Inference module...")
    engine = VikramadithyaInferenceEngine()

    # Test KNN search with LTT 9779 b parameters (ultra-short-period)
    neighbors = engine.find_nearest_neighbors(
        period=0.792,
        transit_depth_ppm=7500.0,
        teff=5443,
    )
    print("\nKNN Results (target: LTT 9779 b — 0.792d, 7500ppm, 5443K):")
    for n in neighbors:
        print(f"  - {n['name']}  |  P={n['period']:.3f}d  Depth={n['depth']:.0f}ppm  "
              f"Teff={n['teff']:.0f}K  |  Similarity: {n['similarity_pct']:.1f}%")

    # Test with a long-period target (Kepler-452b-like)
    neighbors2 = engine.find_nearest_neighbors(
        period=384.84,
        transit_depth_ppm=4000.0,
        teff=5757,
    )
    print("\nKNN Results (target: Kepler-452b — 384.84d, 4000ppm, 5757K):")
    for n in neighbors2:
        print(f"  - {n['name']}  |  P={n['period']:.3f}d  Depth={n['depth']:.0f}ppm  "
              f"Teff={n['teff']:.0f}K  |  Similarity: {n['similarity_pct']:.1f}%")