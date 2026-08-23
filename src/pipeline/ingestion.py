import os
import asyncio
import logging
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightkurve as lk
from astroquery.mast import Observations

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ingestion")

class TESSIngestionEngine:
    """
    High-throughput, asynchronous target processing engine for NASA TESS telemetry.
    Integrates astroquery.mast to dynamically retrieve/stream data, applies rolling-median
    detrending using lightkurve, phase-folds the signal, and generates 2D transit morphology plots.
    """
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        # Thread pool executor for running blocking astronomy calls
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=4)

    def close(self) -> None:
        """Release worker threads owned by this ingestion engine."""
        self._executor.shutdown(wait=True)

    def _query_mast_and_download(self, tic_id: int) -> lk.LightCurveCollection:
        """
        Synchronous worker function to query MAST and load the TESS lightcurves.
        Uses astroquery under the hood via lightkurve search.
        """
        logger.info(f"Querying MAST API for TIC {tic_id}...")
        # Explicitly use astroquery to query metadata first to demonstrate integration and fetch properties
        try:
            target_name = f"TIC {tic_id}"
            obs_table = Observations.query_criteria(objectname=target_name, project="TESS", obs_collection="TESS")
            logger.info(f"MAST query returned {len(obs_table)} observation sectors for TIC {tic_id}")
        except Exception as e:
            logger.warning(f"Direct astroquery.mast metadata query failed: {e}. Falling back to standard lightkurve search.")

        # Search lightcurves using lightkurve (which queries MAST)
        # Search for SPOC or TESS-SPOC produced standard lightcurves to ensure high-cadence and proper format
        search_result = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC")
        if len(search_result) == 0:
            logger.info("SPOC author search empty, searching standard TESS lightcurves...")
            search_result = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
            
        if len(search_result) == 0:
            raise ValueError(f"No TESS lightcurves found for TIC {tic_id}")
        
        # Download files. This is dynamic on-demand streaming
        logger.info(f"Downloading/streaming {len(search_result)} lightcurve files for TIC {tic_id}...")
        lc_collection = search_result.download_all(download_dir=self.cache_dir)
        if lc_collection is None or len(lc_collection) == 0:
            raise ValueError(f"Failed to download lightcurves for TIC {tic_id}")
        return lc_collection

    def _fetch_metadata_sync(self, tic_id: int) -> Dict[str, Any]:
        """
        Queries MAST to fetch physical metadata (Teff, Radius, Mass, etc.) if available.
        Returns a dictionary of metadata.
        """
        metadata = {
            "tic_id": tic_id,
            "mass": 1.0,        # Solar mass default
            "radius": 1.0,      # Solar radii default
            "teff": 5778.0,     # Kelvin (Solar Teff)
            "dec": 0.0,
            "ra": 0.0,
            "success": False
        }
        try:
            target_name = f"TIC {tic_id}"
            obs_table = Observations.query_criteria(objectname=target_name, project="TESS", obs_collection="TESS")
            if len(obs_table) > 0:
                # Retrieve standard columns if present, otherwise fallback
                row = obs_table[0]
                if "target_classification" in obs_table.colnames:
                    metadata["classification"] = str(row["target_classification"])
                if "s_dec" in obs_table.colnames:
                    metadata["dec"] = float(row["s_dec"])
                if "s_ra" in obs_table.colnames:
                    metadata["ra"] = float(row["s_ra"])
                
                # Fetch target properties from MAST Catalog
                from astroquery.mast import Catalogs
                tic_catalog = Catalogs.query_object(target_name, catalog="TIC", radius=0.01)
                if len(tic_catalog) > 0:
                    tic_row = tic_catalog[0]
                    if "mass" in tic_catalog.colnames and not np.isnan(tic_row["mass"]):
                        metadata["mass"] = float(tic_row["mass"])
                    if "rad" in tic_catalog.colnames and not np.isnan(tic_row["rad"]):
                        metadata["radius"] = float(tic_row["rad"])
                    if "Teff" in tic_catalog.colnames and not np.isnan(tic_row["Teff"]):
                        metadata["teff"] = float(tic_row["Teff"])
                    metadata["success"] = True
                    logger.info(f"Successfully fetched TIC {tic_id} physical properties: Mass={metadata['mass']}, Rad={metadata['radius']}, Teff={metadata['teff']}")
        except Exception as e:
            logger.warning(f"Could not retrieve precise physical metadata from MAST for TIC {tic_id}: {e}")
        return metadata

    async def fetch_target_data(self, tic_id: int) -> Tuple[lk.LightCurveCollection, Dict[str, Any]]:
        """
        Asynchronously fetches lightcurves and metadata for a given TIC ID.
        """
        loop = asyncio.get_running_loop()
        # Fetch lightcurve collection and metadata concurrently
        lc_task = loop.run_in_executor(self._executor, self._query_mast_and_download, tic_id)
        meta_task = loop.run_in_executor(self._executor, self._fetch_metadata_sync, tic_id)
        
        lc_collection, metadata = await asyncio.gather(lc_task, meta_task)
        return lc_collection, metadata

    def detrend_and_fold(self, lc_collection: Any, orbital_period: float, window_length: int = 101) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        1. Stitches sectors of TESS data together.
        2. Applies rolling-median detrending to normalize baseline flux to 1.0.
        3. Folds the light curve by the specified orbital period.
        4. Estimates basic transit parameters: Depth, Duration, and SNR.
        """
        logger.info(f"Stitching sectors. Total sectors: {len(lc_collection)}")
        # Stitch all sectors together
        lc = lc_collection.stitch()
        lc = lc.remove_nans()

        logger.info(f"Applying rolling-window median detrending with window_length={window_length}")
        # Custom rolling-window median detrending using pandas
        df = lc.to_pandas()
        # Ensure correct window (must be odd, or just use integer)
        w_len = int(window_length)
        if w_len % 2 == 0:
            w_len += 1
            
        # Compute rolling median
        rolling_median = df['flux'].rolling(window=w_len, center=True, min_periods=1).median()
        normalized_flux = df['flux'].values / rolling_median.values
        
        # Write back to lightcurve object
        lc.flux = normalized_flux
        # Normalize the errors too by the rolling median to keep consistent
        if hasattr(lc, 'flux_err') and lc.flux_err is not None:
            lc.flux_err = lc.flux_err.value / rolling_median.values

        logger.info(f"Phase-folding light curve with orbital period P = {orbital_period} days")
        folded_lc = lc.fold(period=orbital_period)
        
        # --- Parameter Estimation ---
        # Estimate Transit Depth (ppm)
        flux_vals = folded_lc.flux.value
        # Use 1st percentile to avoid outliers for depth
        depth_val = 1.0 - np.percentile(flux_vals, 1.0)
        depth_ppm = depth_val * 1e6
        
        # Estimate SNR: Depth / Standard Deviation of flux (noise)
        noise = np.std(flux_vals)
        snr = depth_val / noise if noise > 0 else 0.0
        
        # Estimate Duration (simple fraction of period where flux is below threshold)
        threshold = 1.0 - (depth_val * 0.5)
        duration_mask = flux_vals < threshold
        duration_frac = np.sum(duration_mask) / len(flux_vals)
        duration_hours = duration_frac * orbital_period * 24.0
        
        est_params = {
            "depth_ppm": float(depth_ppm),
            "snr": float(snr),
            "duration_hours": float(duration_hours)
        }
        
        return lc, folded_lc, est_params

    def export_morphology_plot(self, folded_lc: lk.FoldedLightCurve, output_path: str) -> str:
        """
        Exports a clean 2D transit morphology scatter plot (224x224 pixels, grayscale, normalized)
        saved locally as a temporary PNG image.
        """
        logger.info(f"Generating 2D transit morphology image at {output_path}")
        
        # Filter NaNs just in case
        time_vals = folded_lc.time.value
        flux_vals = folded_lc.flux.value
        mask = ~np.isnan(time_vals) & ~np.isnan(flux_vals)
        time_vals = time_vals[mask]
        flux_vals = flux_vals[mask]

        # Use matplotlib to draw 224x224 pixels scatter plot
        fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1]) # Cover entire canvas
        ax.axis('off') # No borders, axes, ticks
        
        # Draw scatter plot. We can use black dots on white background (grayscale)
        ax.scatter(time_vals, flux_vals, s=0.2, c='black', alpha=0.8)
        
        # Dynamic limits to focus on the points
        if len(time_vals) > 0:
            ax.set_xlim(np.min(time_vals), np.max(time_vals))
            # Zoom slightly to show shape nicely; clip outliers to avoid empty plots
            p5, p95 = np.percentile(flux_vals, [1, 99])
            span = p95 - p5 if p95 > p5 else 0.1
            ax.set_ylim(p5 - 0.2 * span, p95 + 0.2 * span)
        
        # Save image as PNG
        plt.savefig(output_path, dpi=100, facecolor='white')
        plt.close(fig)
        logger.info(f"Morphology image saved successfully to {output_path}")
        return output_path

async def run_pipeline_for_target(tic_id: int, orbital_period: float, window_length: int, output_img_path: str) -> Dict[str, Any]:
    """
    Main async entry point to run ingestion, detrending, folding, and plotting for a TIC ID.
    """
    engine = TESSIngestionEngine()
    try:
        lc_collection, metadata = await engine.fetch_target_data(tic_id)
        lc, folded_lc, est_params = engine.detrend_and_fold(lc_collection, orbital_period, window_length)
        img_path = engine.export_morphology_plot(folded_lc, output_img_path)

        # Extract some simple metrics for display
        metrics = {
            "metadata": metadata,
            "est_params": est_params,
            "raw_points": len(lc),
            "folded_points": len(folded_lc),
            "img_path": img_path,
            "folded_time": folded_lc.time.value.tolist(),
            "folded_flux": folded_lc.flux.value.tolist(),
            "success": True
        }
        return metrics
    finally:
        engine.close()

if __name__ == "__main__":
    # Quick test execution
    async def main():
        # Let's test with a known exoplanet host, TIC 375506058 (TOI-175 / LTT 9779)
        tic_id = 375506058
        period = 0.79207  # LTT 9779 b period
        import tempfile
        out_file = os.path.join(tempfile.gettempdir(), f"tess_tic_{tic_id}.png")
        print(f"Testing pipeline for TIC {tic_id}...")
        try:
            res = await run_pipeline_for_target(tic_id, period, 101, out_file)
            print("Successfully executed pipeline! Metadata:")
            print(res["metadata"])
            print(f"Output image location: {res['img_path']}")
        except Exception as e:
            print(f"Error during ingestion test: {e}")

    asyncio.run(main())
