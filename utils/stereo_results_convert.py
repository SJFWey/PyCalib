import json
import numpy as np
import cv2
from pathlib import Path


def convert_stereo_report_to_depth_params(
    stereo_report_path: str = "pycalib/results/stereo_calib_report.json",
    output_path: str = "pycalib/configs/cam_params_depth_map_stereo.json",
):
    """
    Converts stereo calibration results to the format used for depth map calculation.

    Args:
        stereo_report_path: Path to the stereo calibration report JSON file.
        output_path: Path where the converted parameters will be saved.
    """
    report_path = Path(stereo_report_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "r") as f:
        stereo_data = json.load(f)

    # --- Process Right Camera (maps to "cam" in output) ---
    right_cam_data = stereo_data["right_camera"]
    right_intr = right_cam_data["intrinsics"]
    right_extr = right_cam_data["extrinsics"]
    right_res = stereo_data["metadata"]["right_resolution"]

    right_K = np.array(right_intr["K"])
    right_dist = np.array(right_intr["dist_coeffs"])
    right_rvec = np.array(right_extr["rvec"]).flatten()
    right_tvec = np.array(right_extr["tvec"]).flatten().reshape(3, 1)
    right_R, _ = cv2.Rodrigues(right_rvec)
    right_Rt = np.hstack((right_R, right_tvec))
    right_undist = -right_dist

    cam_params = {
        "K": right_K.tolist(),
        "Rt": right_Rt.tolist(),
        "distortion": right_dist.tolist(),
        "resolution": right_res,
        "undistortion": right_undist.tolist(),
    }

    # --- Process Left Camera (maps to "projector" in output) ---
    left_cam_data = stereo_data["left_camera"]
    left_intr = left_cam_data["intrinsics"]
    left_extr = left_cam_data["extrinsics"]
    left_res = stereo_data["metadata"]["left_resolution"]

    left_K = np.array(left_intr["K"])
    left_dist = np.array(left_intr["dist_coeffs"])
    left_rvec = np.array(left_extr["rvec"]).flatten()
    left_tvec = np.array(left_extr["tvec"]).flatten().reshape(3, 1)
    left_R, _ = cv2.Rodrigues(left_rvec)
    left_Rt = np.hstack((left_R, left_tvec))
    left_undist = -left_dist

    projector_params = {
        "K": left_K.tolist(),
        "Rt": left_Rt.tolist(),
        "distortion": left_dist.tolist(),
        "resolution": left_res,
        "undistortion": left_undist.tolist(),
    }

    # --- Combine and Save ---
    output_data = {
        "cams": {"cam": cam_params},
        "projectors": {"projector": projector_params},
    }

    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Converted parameters saved to: {out_path}")


if __name__ == "__main__":
    convert_stereo_report_to_depth_params()
