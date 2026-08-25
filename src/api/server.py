import os
import sys
import tempfile
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.pipeline.ingestion import TESSIngestionEngine

# Import inference engine with graceful fallback if torch isn't available.
# The module itself is importable without torch (classes are defined),
# but instantiation raises ImportError — so we check the availability flag
# from the inference module rather than just the import success.
from src.models.inference import VikramadithyaInferenceEngine, _TORCH_AVAILABLE

INFERENCE_AVAILABLE = _TORCH_AVAILABLE
if not INFERENCE_AVAILABLE:
    VikramadithyaInferenceEngine = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("api")

app = FastAPI(title="INS Vikramadithya API", version="1.0.0")

# Determine frontend HTML path relative to this file
_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
_index_html_path = os.path.join(_frontend_dir, "index.html")

@app.get("/")
async def root():
    """Serve the frontend dashboard."""
    if os.path.exists(_index_html_path):
        return FileResponse(_index_html_path, media_type="text/html")
    return JSONResponse(
        content={
            "service": "INS Vikramadithya",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/health",
            "note": "Frontend index.html not found. Serve src/frontend/index.html directly in your browser."
        }
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (in-memory cache per session)
_ingestion_engine = TESSIngestionEngine()
# Initialize inference engine only if PyTorch is available.
# VikramadithyaInferenceEngine is None when torch is absent (set above).
_inference_engine = None
if INFERENCE_AVAILABLE:
    _inference_engine = VikramadithyaInferenceEngine()

# Session cache
session_cache = {
    "lc_collection": None,
    "metadata": None,
    "lc": None,
    "folded_lc": None,
    "est_params": None,
    "ai_verdict": None,
    "knn_matches": [],
    "morphology_path": None
}

class TargetRequest(BaseModel):
    tic_id: str
    orbital_period: float

class ProcessRequest(BaseModel):
    orbital_period: float
    window_length: int = 101

class OverrideRequest(BaseModel):
    label: str

@app.get("/api/health")
async def health_check():
    return {"status": "operational", "engine": "INS Vikramadithya"}

@app.post("/api/fetch")
async def fetch_target(req: TargetRequest):
    """Fetch TESS lightcurves and metadata from MAST for a given TIC ID."""
    try:
        lc_col, meta = await _ingestion_engine.fetch_target_data(int(req.tic_id))
        session_cache["lc_collection"] = lc_col
        session_cache["metadata"] = meta
        logger.info(f"Successfully fetched TIC {req.tic_id}: {len(lc_col)} sectors")
        return {
            "success": True,
            "tic_id": int(req.tic_id),
            "sectors": len(lc_col),
            "metadata": {
                "mass": meta["mass"],
                "radius": meta["radius"],
                "teff": meta["teff"],
                "dec": meta["dec"],
                "ra": meta["ra"]
            }
        }
    except Exception as e:
        logger.error(f"Fetch error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process")
async def process_data(req: ProcessRequest):
    """Detrend, fold, estimate parameters, and run inference on fetched data."""
    if session_cache["lc_collection"] is None:
        raise HTTPException(status_code=400, detail="No data cached. Fetch target first.")
    
    try:
        # Step 1: Detrend & Fold
        lc, folded_lc, est_params = _ingestion_engine.detrend_and_fold(
            session_cache["lc_collection"], req.orbital_period, req.window_length
        )
        session_cache["lc"] = lc
        session_cache["folded_lc"] = folded_lc
        session_cache["est_params"] = est_params

        # Step 2: Export morphology plot
        temp_dir = tempfile.gettempdir()
        tic_id = session_cache["metadata"]["tic_id"]
        img_path = os.path.join(temp_dir, f"tess_morph_{tic_id}.png")
        _ingestion_engine.export_morphology_plot(folded_lc, img_path)
        session_cache["morphology_path"] = img_path

        # Step 3: Run Swin Transformer inference
        verdict = _inference_engine.evaluate_image(img_path)
        session_cache["ai_verdict"] = verdict

        # Step 4: KNN similarity matching (3D: period, transit_depth_ppm, host teff)
        m = session_cache["metadata"]
        neighbors = _inference_engine.find_nearest_neighbors(
            period=req.orbital_period,
            transit_depth_ppm=est_params["depth_ppm"],
            teff=m.get("teff", 5778.0)
        )
        session_cache["knn_matches"] = neighbors

        # Step 5: Physics based validation
        physics_pass = True
        if session_cache["metadata"] and est_params:
            rs = session_cache["metadata"]["radius"]
            depth = est_params["depth_ppm"] / 1e6
            rp_rearth = rs * np.sqrt(depth) * 109.2
            rp_rjupiter = rp_rearth / 11.2
            if rp_rjupiter > 2.0:
                physics_pass = False

        planet_conf = verdict["planet_confidence"] * 100
        snr_boost = min(est_params["snr"] / 10.0, 1.0)
        reliability = (planet_conf * 0.7) + (snr_boost * 30.0)
        if not physics_pass:
            reliability *= 0.1

        folded_data = {
            "time": folded_lc.time.value.tolist(),
            "flux": folded_lc.flux.value.tolist()
        }

        return {
            "success": True,
            "est_params": {
                "depth_ppm": est_params["depth_ppm"],
                "duration_hours": est_params["duration_hours"],
                "snr": est_params["snr"]
            },
            "ai_verdict": {
                "planet_confidence": planet_conf,
                "false_positive_confidence": verdict["false_positive_confidence"] * 100,
                "reliability_score": reliability,
                "planet_detected": bool(planet_conf > 50.0 and physics_pass),
                "physics_valid": bool(physics_pass)
            },
            "knn_matches": [
                {
                    "name": m["name"],
                    "type": m["type"],
                    "similarity": m["similarity_pct"] / 100.0,  # decimal for frontend rendering
                    "similarity_pct": m["similarity_pct"],
                    "period": m["period"],
                    "depth_ppm": m["depth"],
                    "teff": m["teff"],
                    "discovery_year": m["discovery_year"]
                }
                for m in neighbors
            ],
            "folded_data": folded_data
        }

    except Exception as e:
        logger.error(f"Process error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/morphology")
async def get_morphology():
    """Serve the morphology plot image."""
    if session_cache["morphology_path"] is None or not os.path.exists(session_cache["morphology_path"]):
        raise HTTPException(status_code=404, detail="No morphology image available.")
    return FileResponse(session_cache["morphology_path"], media_type="image/png")

@app.get("/api/state")
async def get_state():
    """Return current session state summary."""
    return {
        "metadata": session_cache.get("metadata"),
        "est_params": session_cache.get("est_params"),
        "ai_verdict": session_cache.get("ai_verdict"),
        "knn_matches": session_cache.get("knn_matches"),
        "has_folded": session_cache.get("folded_lc") is not None
    }

@app.post("/api/override")
async def set_override(req: OverrideRequest):
    """Set manual override label."""
    session_cache["override_label"] = req.label
    return {"success": True, "override": req.label}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")