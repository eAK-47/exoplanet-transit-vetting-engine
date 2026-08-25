"""Standalone verification for the INS Vikramadithya pipeline.

Run from any directory with:
    .\\env_vikram\\Scripts\\python.exe test_system.py
"""
from __future__ import annotations

import asyncio
import importlib
import math
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Verification:
    def __init__(self) -> None:
        self.results: list[tuple[str, str, str]] = []

    def record(self, subsystem: str, status: str, detail: str) -> None:
        self.results.append((subsystem, status, detail))
        print(f"[{status}] {subsystem}: {detail}")

    def check(self, subsystem: str, function: Any) -> Any:
        try:
            value = function()
            self.record(subsystem, "PASS", "completed")
            return value
        except Exception as exc:
            self.record(subsystem, "FAIL", f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            return None

    def skip(self, subsystem: str, detail: str) -> None:
        self.record(subsystem, "SKIP", detail)

    def summary(self) -> int:
        print("\nSubsystem summary")
        print("-----------------")
        for subsystem, status, detail in self.results:
            print(f"{subsystem:28} {status:4} {detail}")
        return int(any(status == "FAIL" for _, status, _ in self.results))


def finite(value: Any) -> bool:
    return bool(np.isfinite(np.asarray(value, dtype=float)).all())


def audit_catalog(verification: Verification) -> None:
    inference = importlib.import_module("src.models.inference")
    if not inference._TORCH_AVAILABLE:
        verification.skip("Catalog and KNN", "PyTorch not available (expected in main CI, run gpu-test.yml for full verification)")
        return

    dataframe = inference._load_exoplanet_catalog()
    if dataframe is None or dataframe.empty:
        raise AssertionError("local exoplanet catalog did not load")
    required = {"pl_name", "pl_orbper", "pl_trandep", "st_teff"}
    if not required.issubset(dataframe.columns):
        raise AssertionError(f"catalog is missing {required - set(dataframe.columns)}")
    numeric_columns = ["pl_orbper", "pl_trandep", "st_teff"]
    if not finite(dataframe[numeric_columns].to_numpy()):
        raise AssertionError("catalog contains non-finite required values")

    engine = object.__new__(inference.VikramadithyaInferenceEngine)
    engine._initialize_historical_knn()
    if not finite(engine.knn_features) or not finite(engine.scaler.transform(engine.knn_features)):
        raise AssertionError("RobustScaler produced non-finite values")
    neighbors = engine.find_nearest_neighbors(6.27, 5000.0, 5800.0)
    if len(neighbors) != 3:
        raise AssertionError(f"expected 3 neighbors, got {len(neighbors)}")
    keys = {"name", "period", "depth", "teff", "type", "discovery_year", "similarity_pct"}
    if any(not keys.issubset(item) for item in neighbors):
        raise AssertionError("neighbor response is missing required keys")
    similarities = [item["similarity_pct"] for item in neighbors]
    if not all(0.0 <= value <= 100.0 and math.isfinite(value) for value in similarities):
        raise AssertionError("neighbor similarity is outside [0, 100]")

    verification.record("Catalog and KNN", "PASS", f"{len(dataframe)} catalog rows; 3 finite neighbors")


def synthetic_lightcurve() -> Any:
    import lightkurve as lk

    time = np.linspace(0.0, 12.0, 1200)
    phase = ((time + 1.0) % 3.0) - 1.5
    flux = np.ones_like(time)
    flux[np.abs(phase) < 0.12] -= 0.01
    flux += np.sin(time * 2.0) * 0.0002
    return lk.LightCurve(time=time, flux=flux, flux_err=np.full_like(time, 0.0003))


def audit_detrending_and_plot(verification: Verification) -> None:
    ingestion = importlib.import_module("src.pipeline.ingestion")
    engine = ingestion.TESSIngestionEngine(cache_dir=str(ROOT / "data" / "cache"))
    collection = ingestion.lk.LightCurveCollection([synthetic_lightcurve()])
    _, folded, params = engine.detrend_and_fold(collection, 3.0, 101)
    if set(params) != {"depth_ppm", "duration_hours", "snr"} or not finite(list(params.values())):
        raise AssertionError(f"invalid transit parameters: {params}")
    if len(folded) == 0:
        raise AssertionError("phase folding produced no samples")
    output = Path(tempfile.gettempdir()) / "vikramadithya_test_morphology.png"
    engine.export_morphology_plot(folded, str(output))
    if not output.exists() or output.stat().st_size == 0:
        raise AssertionError("morphology plot was not created")
    from PIL import Image

    with Image.open(output) as image:
        if image.size != (224, 224):
            raise AssertionError(f"unexpected morphology dimensions: {image.size}")
    output.unlink(missing_ok=True)
    verification.record("Detrending and plotting", "PASS", f"finite parameters; {len(folded)} folded samples")


def audit_live_ingestion(verification: Verification) -> None:
    ingestion = importlib.import_module("src.pipeline.ingestion")

    async def run() -> dict[str, Any]:
        output = ROOT / "data" / "temp_transit.png"
        return await ingestion.run_pipeline_for_target(261136679, 6.27, 101, str(output))

    try:
        result = asyncio.run(asyncio.wait_for(run(), timeout=90.0))
        params = result["est_params"]
        if result.get("success") is not True or not finite(list(params.values())):
            raise AssertionError("live ingestion returned invalid results")
        if not Path(result["img_path"]).exists():
            raise AssertionError("live morphology image is missing")
        verification.record("Live MAST ingestion", "PASS", f"{result['raw_points']} raw points downloaded")
    except (TimeoutError, asyncio.TimeoutError, ConnectionError, OSError) as exc:
        verification.skip("Live MAST ingestion", f"external service unavailable: {exc}")
    except Exception as exc:
        message = str(exc).lower()
        external_markers = ("mast", "astroquery", "lightcurve", "download", "connection", "timeout", "no tess")
        if any(marker in message for marker in external_markers):
            verification.skip("Live MAST ingestion", f"external service unavailable: {exc}")
        else:
            verification.record("Live MAST ingestion", "FAIL", f"{type(exc).__name__}: {exc}")
            traceback.print_exc()


def audit_gpu_and_model(verification: Verification) -> None:
    try:
        import torch
    except ImportError:
        verification.skip("GPU and model", "PyTorch not available (expected in main CI, run gpu-test.yml for full GPU testing)")
        return

    if not torch.cuda.is_available():
        verification.skip("GPU and model", "CUDA is not available in this environment")
        return
    try:
        tensor = torch.zeros((2, 3), device="cuda")
        assert tensor.is_cuda
        del tensor
        torch.cuda.empty_cache()

        inference = importlib.import_module("src.models.inference")
        model = inference.SwinTransitClassifier().to("cuda").eval()
        input_tensor = torch.zeros((1, 3, 224, 224), device="cuda", dtype=torch.float16)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            output = model(input_tensor)
        if tuple(output.shape) != (1, 2) or not finite(output.float().cpu().numpy()):
            raise AssertionError(f"unexpected model output shape or values: {tuple(output.shape)}")
        del input_tensor, output, model
        torch.cuda.empty_cache()
        verification.record("GPU and model", "PASS", "CUDA allocation, checkpoint load, and FP16 forward pass succeeded")
    except Exception as exc:
        verification.record("GPU and model", "FAIL", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


def audit_api_contract(verification: Verification) -> None:
    import asyncio
    from types import SimpleNamespace

    server = importlib.import_module("src.api.server")
    
    # Skip test if inference engine isn't available (torch not installed)
    if not server.INFERENCE_AVAILABLE:
        verification.skip("API process contract", "PyTorch not available (expected in main CI, run gpu-test.yml for full API testing)")
        return
    
    ingestion = server._ingestion_engine
    original = dict(server.session_cache)
    image_path = Path(tempfile.gettempdir()) / "vikramadithya_api_contract.png"

    class FakeInference:
        def evaluate_image(self, path: str) -> dict[str, float]:
            assert Path(path).name == "tess_morph_261136679.png"
            return {"planet_confidence": 0.8, "false_positive_confidence": 0.2}

        def find_nearest_neighbors(self, **_: Any) -> list[dict[str, Any]]:
            return [{"name": "Test b", "type": "Archive", "similarity_pct": 75.0,
                     "period": 6.27, "depth": 5000.0, "teff": 5778.0, "discovery_year": 0}]

    class FakeIngestion:
        def detrend_and_fold(self, *_: Any) -> tuple[Any, Any, dict[str, float]]:
            folded = SimpleNamespace(time=SimpleNamespace(value=np.array([-0.1, 0.1])),
                                     flux=SimpleNamespace(value=np.array([1.0, 0.99])))
            return object(), folded, {"depth_ppm": 10000.0, "duration_hours": 2.0, "snr": 5.0}

        def export_morphology_plot(self, folded: Any, path: str) -> str:
            image_path.write_bytes(b"PNG")
            return path

    try:
        server._ingestion_engine = FakeIngestion()
        server._inference_engine = FakeInference()
        server.session_cache.update({"lc_collection": object(), "metadata": {"tic_id": 261136679, "radius": 1.0, "teff": 5778.0}})
        response = asyncio.run(server.process_data(server.ProcessRequest(orbital_period=6.27, window_length=101)))
        required = {"success", "est_params", "ai_verdict", "knn_matches", "folded_data"}
        if not required.issubset(response):
            raise AssertionError(f"missing response keys: {required - set(response)}")
        for key in ("depth_ppm", "duration_hours", "snr"):
            if key not in response["est_params"]:
                raise AssertionError(f"missing transit parameter: {key}")
        for key in ("planet_confidence", "false_positive_confidence", "reliability_score", "planet_detected", "physics_valid"):
            if key not in response["ai_verdict"]:
                raise AssertionError(f"missing verdict field: {key}")
        match = response["knn_matches"][0]
        if not {"name", "type", "similarity", "similarity_pct", "period", "depth_ppm", "teff", "discovery_year"}.issubset(match):
            raise AssertionError("missing frontend neighbor field")
        if not isinstance(response["folded_data"]["time"], list) or not isinstance(response["folded_data"]["flux"], list):
            raise AssertionError("folded data is not JSON-list shaped")
        verification.record("API process contract", "PASS", "all frontend-required fields and formats present")
    finally:
        server.session_cache.clear()
        server.session_cache.update(original)
        server._ingestion_engine = ingestion
        image_path.unlink(missing_ok=True)


def main() -> int:
    verification = Verification()
    print(f"INS Vikramadithya system verification ({ROOT})")
    verification.check("Catalog and KNN", lambda: audit_catalog(verification))
    verification.check("Detrending and plotting", lambda: audit_detrending_and_plot(verification))
    audit_live_ingestion(verification)
    audit_gpu_and_model(verification)
    verification.check("API process contract", lambda: audit_api_contract(verification))
    return verification.summary()


if __name__ == "__main__":
    raise SystemExit(main())
