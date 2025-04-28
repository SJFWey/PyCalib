import os
import pickle
from collections import defaultdict

import numpy as np

from pycalib.optimization.optimizer_configs import (
    OptimizerParams,
    StereoCalibrationFlags,
)
from pycalib.optimization.optimizer_stereo import StereoOptimizer


def load_feature_data(feature_data_path):
    calib_data = pickle.load(open(feature_data_path, "rb"))
    feature_data = defaultdict(dict)
    if "image_points" in calib_data.keys() and "object_points" in calib_data.keys():
        feature_data["image_points"] = calib_data["image_points"]
        feature_data["object_points"] = calib_data["object_points"]
    else:
        feature_data["image_points"] = defaultdict(dict)
        feature_data["object_points"] = defaultdict(dict)
        for frame in calib_data:
            feature_data["image_points"][frame] = calib_data[frame]["imagePoints"]
            feature_data["object_points"][frame] = calib_data[frame]["objectPoints"]

    return feature_data


def load_calib_data(calib_file_path):
    """Load calibration data from pickle file."""
    with open(calib_file_path, "rb") as f:
        return pickle.load(f)


def calib_stereo(
    left_feature_data,
    right_feature_data,
    left_calib_file,
    right_calib_file,
    flags,
    opt_params,
):
    left_feature_data = load_feature_data(left_feature_data)
    right_feature_data = load_feature_data(right_feature_data)

    left_calib_data = load_calib_data(left_calib_file)
    right_calib_data = load_calib_data(right_calib_file)

    stereo_optimizer = StereoOptimizer(
        left_resolution=(1280, 720),
        right_resolution=(768, 768),
        left_feature_data=left_feature_data,
        right_feature_data=right_feature_data,
        left_calib_data=left_calib_data,
        right_calib_data=right_calib_data,
        flags=flags,
        optimizer_params=opt_params,
    )

    initial_frames = stereo_optimizer.valid_frames
    print(f"Found {len(initial_frames)} valid common frames initially")

    results = stereo_optimizer._calibrate()

    if results is None:
        raise ValueError("ERROR: Calibration failed")

    print(f"\nFinal RMS error: {results['rms_error_px']:.5f} pixels")

    final_valid_frames = stereo_optimizer.valid_frames
    print(f"After calibration: {len(final_valid_frames)} valid frames remaining")
    if (
        hasattr(stereo_optimizer, "discarded_frames")
        and stereo_optimizer.discarded_frames
    ):
        print(f"Frames removed as outliers: {len(stereo_optimizer.discarded_frames)}")

    print("\nFinal relative pose:")
    rel_pose = results["relative_pose"]
    print(
        f"Rotation (deg): {rel_pose['euler_angles_deg'][0]:.2f}, {rel_pose['euler_angles_deg'][1]:.2f}, {rel_pose['euler_angles_deg'][2]:.2f}"
    )
    tvec = np.asarray(rel_pose["tvec"]).flatten()
    print(f"Translation: [{tvec[0]:.2f}, {tvec[1]:.2f}, {tvec[2]:.2f}]")


if __name__ == "__main__":
    left_calib_file = "pycalib/data/cache/calib_result_Projector.pkl"
    right_calib_file = "pycalib/data/cache/calib_result_Camera.pkl"

    left_feature_data = "pycalib/data/cache/filtered_proj_feature_data.pkl"
    right_feature_data = "pycalib/data/cache/filtered_cam_feature_data.pkl"

    assert os.path.exists(left_calib_file), (
        f"Left calibration file not found: {left_calib_file}"
    )
    assert os.path.exists(right_calib_file), (
        f"Right calibration file not found: {right_calib_file}"
    )

    flags = StereoCalibrationFlags(
        estimate_left_intrinsics=True,
        estimate_right_intrinsics=True,
    )

    opt_params = OptimizerParams(
        max_iter=50, ftol=1e-4, xtol=1e-4, gtol=1e-4, opt_method="lm"
    )

    print("=== Start Stereo Calibration ===")
    calib_stereo(
        left_feature_data,
        right_feature_data,
        left_calib_file,
        right_calib_file,
        flags,
        opt_params,
    )
