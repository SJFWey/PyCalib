import json
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from pycalib.feature_processing.feature_processing_utils import rvec_to_euler
from pycalib.optimization.optimizer_configs import (
    Extrinsics,
    OptimizerFlags,
    OptimizerParams,
    ParamsGuess,
)
from pycalib.optimization.optimizer_single import Optimizer

color_divide = "\033[1;32;40m"
color_end = "\033[0m"


def calib_single(
    cam_name: str,
    feature_data: dict[dict],
    frame_to_exclude: Optional[list] = None,
    intris_guess: ParamsGuess = None,
    flags: OptimizerFlags = None,
    opt_params: OptimizerParams = None,
) -> dict:
    feature_data = preprocess_feature_data(feature_data)

    if frame_to_exclude:
        feature_data = {
            k: v for k, v in feature_data.items() if k not in frame_to_exclude
        }

    print(f"\n=== Starting calibration for {cam_name} ===")
    intr_optimizer = Optimizer(
        cam_name,
        feature_data,
        params_guess=intris_guess,
        optimizer_params=opt_params,
        flags=flags,
    )

    calib_results = intr_optimizer._calibrate()

    output_dir = "pycalib/data/cache"
    os.makedirs(output_dir, exist_ok=True)
    pkl_output_file = f"{output_dir}/calib_result_{cam_name}.pkl"
    with open(pkl_output_file, "wb") as f:
        pickle.dump(calib_results, f)

    organized_results = save_calib_results(
        cam_name, intris_guess.image_size, calib_results
    )
    print_calib_results(organized_results)

    return organized_results


def preprocess_feature_data(feature_data):
    image_points = {}
    object_points = {}

    for folder_id, data in feature_data.items():
        if "imagePoints" in data and "objectPoints" in data:
            image_points[folder_id] = data["imagePoints"]
            object_points[folder_id] = data["objectPoints"]

    return {"image_points": image_points, "object_points": object_points}


def save_calib_results(cam_name: str, image_size: tuple, results: dict):
    output_dir = "pycalib/results"
    os.makedirs(output_dir, exist_ok=True)
    json_output_file = f"{output_dir}/{cam_name}_calib_report.json"

    if "intrinsics" in results and hasattr(results["intrinsics"], "fx"):
        intr = results["intrinsics"]
        fx, fy, cx, cy = intr.fx, intr.fy, intr.cx, intr.cy
        dist_coeffs = intr.dist_coeffs.tolist()
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    else:
        raise ValueError(
            "Intrinsics data not found or in unexpected format in results."
        )

    json_results = {
        "cam_name": cam_name,
        "resolution": image_size,
        "intrinsics": {
            "K": K.tolist(),
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "dist_coeffs": dist_coeffs,
        },
        "extrinsics": {},
        "errors": {},
    }

    if "extrinsics" in results and isinstance(results["extrinsics"], Extrinsics):
        extr = results["extrinsics"]
        rvec = extr.rvec
        tvec = extr.tvec
        rvec_np = np.asarray(rvec).flatten()
        tvec_np = np.asarray(tvec).flatten()
        euler_angles = rvec_to_euler(rvec_np)
        json_results["extrinsics"] = {
            "rvec": rvec_np.tolist(),
            "tvec": tvec_np.tolist(),
            "euler_angles": [float(a) for a in euler_angles],
        }

    if "errors" in results:
        error_report = results["errors"]
        json_results["errors"] = {
            k: (
                float(v)
                if isinstance(v, (np.number, float, int)) and not np.isnan(v)
                else None
            )
            for k, v in error_report.items()
            if k != "per_frame_rms_px"
        }
        json_results["errors"]["per_frame_rms_px"] = {
            str(frame_id): (
                float(rms)
                if isinstance(rms, (np.number, float, int)) and not np.isnan(rms)
                else None
            )
            for frame_id, rms in error_report.get("per_frame_rms_px", {}).items()
        }

    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return None if np.isnan(obj) else float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(i) for i in obj]
        else:
            return obj

    json_results_serializable = convert_numpy_types(json_results)

    with open(json_output_file, "w") as f:
        json.dump(json_results_serializable, f, indent=2)

    return json_results_serializable


def print_calib_results(results: dict):
    """
    Prints a summary of the calibration results to the console.

    Args:
        results: Organized calibration results dictionary (after JSON saving/loading).
    """
    print(
        f"\n{color_divide}==== {results['cam_name']} Calibration Results ===={color_end}"
    )
    if "intrinsics" in results:
        intr = results["intrinsics"]
        print(
            f" K (Intrinsic Matrix):\n"
            f"   {intr.get('fx', 'N/A'):.2f}  {0:.2f}  {intr.get('cx', 'N/A'):.2f}\n"
            f"   {0:.2f}  {intr.get('fy', 'N/A'):.2f}  {intr.get('cy', 'N/A'):.2f}\n"
            f"   {0:.2f}  {0:.2f}  {1:.2f}"
        )
        print("\n Distortion Coefficients (k1, k2, p1, p2, k3):")
        dist_coeffs = intr.get("dist_coeffs", [])
        print(
            "   "
            + (
                ", ".join([f"{coeff:.4f}" for coeff in dist_coeffs])
                if dist_coeffs
                else "N/A"
            )
        )
    else:
        print(" Intrinsic parameters not available.")

    if "extrinsics" in results:
        extr = results["extrinsics"]
        rvec = extr.get("rvec", ["N/A"] * 3)
        tvec = extr.get("tvec", ["N/A"] * 3)
        euler = extr.get("euler_angles", ["N/A"] * 3)
        print(
            f" rvec: [{', '.join(f'{v:.3f}' if isinstance(v, float) else str(v) for v in rvec)}]"
        )
        print(
            f" tvec: [{', '.join(f'{v:.3f}' if isinstance(v, float) else str(v) for v in tvec)}]"
        )
        print(
            f" rotation (deg): [{', '.join(f'{v:.2f}' if isinstance(v, float) else str(v) for v in euler)}]"
        )

    if "errors" in results and results["errors"]:
        stats = results["errors"]
        print("\n-- Reprojection Error Statistics --")
        print(
            f" Mean Error: {stats.get('mean_px', 'N/A'):.3f} px"
            if stats.get("mean_px") is not None
            else " Mean Error: N/A px"
        )
        print(
            f" RMS Error: {stats.get('rms_px', 'N/A'):.3f} px"
            if stats.get("rms_px") is not None
            else " RMS Error: N/A px"
        )
        print(
            f" Median Error: {stats.get('median_px', 'N/A'):.3f} px"
            if stats.get("median_px") is not None
            else " Median Error: N/A px"
        )
        print(
            f" Min Error: {stats.get('min_px', 'N/A'):.3f} px"
            if stats.get("min_px") is not None
            else " Min Error: N/A px"
        )
        print(
            f" Max Error: {stats.get('max_px', 'N/A'):.3f} px"
            if stats.get("max_px") is not None
            else " Max Error: N/A px"
        )
    else:
        print("\n Error statistics not available.")


if __name__ == "__main__":
    
    use_filtered_data = True
    if use_filtered_data:
        cam_features = Path("pycalib/data/cache/filtered_cam_feature_data.pkl")
    else:
        cam_features = Path("pycalib/data/cache/cam_feature_data.pkl")

    if not cam_features.exists():
        raise FileNotFoundError("Camera feature data file not found")

    with open(cam_features, "rb") as infile:
        cam_feature_data = pickle.load(infile)

    flags_cam = OptimizerFlags(
        estimate_focal=True,
        estimate_principal=True,
        estimate_extrinsics=True,
    )
    intris_guess_cam = ParamsGuess(
        image_size=(1280, 720),
        fx=1000,
        fy=1000,
        cx=640,
        cy=360,
    )

    opt_params_cam = OptimizerParams(
        max_iter=50, opt_method="lm", ftol=1e-4, xtol=1e-4, gtol=1e-4
    )
    results = calib_single(
        "Camera",
        cam_feature_data,
        intris_guess=intris_guess_cam,
        opt_params=opt_params_cam,
        flags=flags_cam,
    )

    #####################################################################################
    if use_filtered_data:
        proj_features = Path("pycalib/data/cache/filtered_proj_feature_data.pkl")
    else:
        proj_features = Path("pycalib/data/cache/proj_feature_data.pkl")

    if not proj_features.exists():
        raise FileNotFoundError("Projector feature data file not found")

    with open(proj_features, "rb") as infile:
        proj_feature_data = pickle.load(infile)

    flags_proj = OptimizerFlags(
        estimate_focal=True,
        estimate_principal=True,
        estimate_extrinsics=True,
    )
    intris_guess_proj = ParamsGuess(
        image_size=(768, 768),
        fx=1300,
        fy=1300,
        cx=384,
        cy=384,
    )

    opt_params_proj = OptimizerParams(
        max_iter=50, opt_method="lm", ftol=1e-6, xtol=1e-6, gtol=1e-6
    )
    results = calib_single(
        "Projector",
        proj_feature_data,
        intris_guess=intris_guess_proj,
        opt_params=opt_params_proj,
        flags=flags_proj,
    )
