import os
import pickle
from pathlib import Path
import json
import cv2
import numpy as np
from pycalib.optimization.optimizer_configs import OptimizerParams
from pycalib.optimization.optimizer_stereo import (
    StereoCalibrationFlags,
    StereoOptimizer,
)
from pycalib.feature_processing.feature_processing_utils import rvec_to_euler
from collections import defaultdict


def align_points_between_frames(left_feature_data, right_feature_data):
    """
    Aligns feature points between left and right cameras based on common object points.
    """
    common_frames = set(left_feature_data.keys()) & set(right_feature_data.keys())
    print(f"Found {len(common_frames)} common frames between cameras")

    aligned_object_points = []
    aligned_left_points = []
    aligned_right_points = []
    valid_frame_indices = []

    for frame_idx in sorted(common_frames):
        left_data = left_feature_data[frame_idx]
        right_data = right_feature_data[frame_idx]

        if (
            "imagePoints" in left_data
            and "objectPoints" in left_data
            and "imagePoints" in right_data
            and "objectPoints" in right_data
        ):
            left_obj_points = np.array(left_data["objectPoints"], dtype=np.float32)
            right_obj_points = np.array(right_data["objectPoints"], dtype=np.float32)
            left_img_points = np.array(left_data["imagePoints"], dtype=np.float32)
            right_img_points = np.array(right_data["imagePoints"], dtype=np.float32)

            # Find common object points between left and right cameras
            common_points = []
            left_indices = []
            right_indices = []

            # For each object point in left camera, find matching one in right camera
            for i, left_point in enumerate(left_obj_points):
                for j, right_point in enumerate(right_obj_points):
                    if np.all(
                        np.isclose(left_point, right_point, rtol=1e-5, atol=1e-5)
                    ):
                        common_points.append(left_point)
                        left_indices.append(i)
                        right_indices.append(j)
                        break

            # If we have enough common points (at least 6), add them to our lists
            if len(common_points) >= 6:
                aligned_object_points.append(np.array(common_points, dtype=np.float32))
                aligned_left_points.append(left_img_points[left_indices])
                aligned_right_points.append(right_img_points[right_indices])
                valid_frame_indices.append(frame_idx)

    print(f"Aligned {len(valid_frame_indices)} frames with common object points")

    if not aligned_object_points:
        raise ValueError("No common object points found between cameras")

    return (
        aligned_object_points,
        aligned_left_points,
        aligned_right_points,
        valid_frame_indices,
    )


def run_cv_stereo(
    output_path="pycalib/results/stereo_calib_report_cv.json",
    left_name="Projector",
    right_name="Camera",
    img_size_left=(1280, 720),
    img_size_right=(1280, 720),
    left_pkl_path=None,
    right_pkl_path=None,
):
    """
    Performs stereo calibration using OpenCV with pre-saved single calibration results.

    Args:
        output_path: Path to save the stereo calibration JSON report.
        left_name: Name of the left camera.
        right_name: Name of the right camera.
        img_size_left: Image size (width, height) for the left camera.
        img_size_right: Image size (width, height) for the right camera.
        flags: Flags for cv2.stereoCalibrate. Default is cv2.CALIB_FIX_INTRINSIC.
        left_pkl_path: Path to the camera calibration result pickle file.
        right_pkl_path: Path to the projector calibration result pickle file.

    Returns:
        dict: A dictionary containing the stereo calibration results, or None if failed.
    """
    # Load organized calibration results
    with open(left_pkl_path, "rb") as f:
        left_calib_data = pickle.load(f)
    with open(right_pkl_path, "rb") as f:
        right_calib_data = pickle.load(f)

    # Extract camera matrices and distortion coefficients
    K_left = np.array(
        [
            [
                left_calib_data["intrinsics"].fx,
                0,
                left_calib_data["intrinsics"].cx,
            ],
            [0, left_calib_data["intrinsics"].fy, left_calib_data["intrinsics"].cy],
            [0, 0, 1],
        ]
    )
    dist_left = np.array(left_calib_data["intrinsics"].dist_coeffs)

    K_right = np.array(
        [
            [
                right_calib_data["intrinsics"].fx,
                0,
                right_calib_data["intrinsics"].cx,
            ],
            [
                0,
                right_calib_data["intrinsics"].fy,
                right_calib_data["intrinsics"].cy,
            ],
            [0, 0, 1],
        ]
    )
    dist_right = np.array(right_calib_data["intrinsics"].dist_coeffs)

    # Get common frames between left and right cameras
    common_frames = set(left_calib_data["valid_frames"]) & set(
        right_calib_data["valid_frames"]
    )

    if len(common_frames) < 3:
        print(
            "Error: Not enough common frames for stereo calibration (minimum 3 required)"
        )
        return None

    # Use align_points_between_frames to get aligned points
    left_feature_data = {
        frame_idx: {
            "imagePoints": left_calib_data["feature_data"]["image_points"][frame_idx],
            "objectPoints": left_calib_data["feature_data"]["object_points"][frame_idx],
        }
        for frame_idx in common_frames
    }

    right_feature_data = {
        frame_idx: {
            "imagePoints": right_calib_data["feature_data"]["image_points"][frame_idx],
            "objectPoints": right_calib_data["feature_data"]["object_points"][
                frame_idx
            ],
        }
        for frame_idx in common_frames
    }

    (
        all_object_points,
        all_left_image_points,
        all_right_image_points,
        valid_frame_indices,
    ) = align_points_between_frames(left_feature_data, right_feature_data)

    discarded_frames = common_frames - set(valid_frame_indices)
    if discarded_frames:
        print(f"Frames removed as outliers: {len(discarded_frames)}")

    if not all_object_points:
        print("Error: No common object points found between cameras")
        return None

    ret, K_left, dist_left, K_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
        all_object_points,
        all_left_image_points,
        all_right_image_points,
        K_left.astype(np.float32),
        dist_left.astype(np.float32),
        K_right.astype(np.float32),
        dist_right.astype(np.float32),
        img_size_left,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS,
    )

    print(f"\nFinal RMS error: {ret:.5f} pixels")

    # Create result dictionary
    calibration_result = {
        "reprojection_error": float(ret),
        "left_camera": {
            "cam_name": left_name,
            "camera_matrix": K_left.tolist(),
            "dist_coeffs": dist_left.flatten().tolist(),
            "image_size": list(img_size_left),
        },
        "right_camera": {
            "cam_name": right_name,
            "camera_matrix": K_right.tolist(),
            "dist_coeffs": dist_right.flatten().tolist(),
            "image_size": list(img_size_right),
        },
        "stereo": {
            "rotation": R.tolist(),
            "translation": T.flatten().tolist(),
            "essential_matrix": E.tolist(),
            "fundamental_matrix": F.tolist(),
        },
        "valid_frame_indices": valid_frame_indices,
        "num_frames_used": len(valid_frame_indices),
    }

    # Save the results
    save_stereo_results(calibration_result, output_path)

    return calibration_result


def save_stereo_results(calibration_result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)

    # Convert rotation matrix to Euler angles (ZYX order)
    R = np.array(calibration_result["stereo"]["rotation"])
    T = np.array(calibration_result["stereo"]["translation"])
    euler_angles = rvec_to_euler(cv2.Rodrigues(R)[0])

    # Add Euler angles to the result
    calibration_result["stereo"]["euler_angles_deg"] = euler_angles

    print("\nFinal relative pose:")
    print(
        f"Rotation (deg): {euler_angles[0]:.2f}, {euler_angles[1]:.2f}, {euler_angles[2]:.2f}"
    )
    print(f"Translation: [{T[0]:.2f}, {T[1]:.2f}, {T[2]:.2f}]")

    with open(output_path, "w") as f:
        json.dump(calibration_result, f, indent=2)

    print(f"Stereo calibration saved to {output_path}")


def run_new_stereo(
    left_calib_file, right_calib_file, left_feature_data=None, right_feature_data=None
):
    assert os.path.exists(left_calib_file), (
        f"Left calibration file not found: {left_calib_file}"
    )
    assert os.path.exists(right_calib_file), (
        f"Right calibration file not found: {right_calib_file}"
    )

    # Load calibration results with the correct method
    left_calib_data = load_calib_data(left_calib_file)
    right_calib_data = load_calib_data(right_calib_file)

    # Use feature data from calibration results if not provided separately
    left_fd = (
        load_feature_data(left_feature_data)
        if left_feature_data
        else left_calib_data["feature_data"]
    )
    right_fd = (
        load_feature_data(right_feature_data)
        if right_feature_data
        else right_calib_data["feature_data"]
    )

    # Configure stereo calibration
    stereo_flags = StereoCalibrationFlags(
        estimate_left_intrinsics=True,
        estimate_right_intrinsics=True,
    )

    stereo_optimizer_params = OptimizerParams(
        max_iter=100, ftol=1e-4, xtol=1e-4, gtol=1e-4, opt_method="lm", verbose=1
    )

    # Initialize stereo optimizer
    stereo_optimizer = StereoOptimizer(
        left_resolution=(1280, 720),
        right_resolution=(768, 768),
        left_feature_data=left_fd,
        right_feature_data=right_fd,
        left_calib_data=left_calib_data,
        right_calib_data=right_calib_data,
        flags=stereo_flags,
        optimizer_params=stereo_optimizer_params,
    )

    # Record initial frames for comparison
    initial_frames = stereo_optimizer.valid_frames
    print(f"Found {len(initial_frames)} valid common frames initially")

    results = stereo_optimizer._calibrate()
    stereo_optimizer.save_results(results)

    if results is None:
        raise ValueError("ERROR: Calibration failed")

    # Print calibration statistics
    print(f"\nFinal RMS error: {results['stats']['rms_error']:.5f} pixels")

    # Print frame information
    final_valid_frames = stereo_optimizer.valid_frames
    print(f"After calibration: {len(final_valid_frames)} valid frames remaining")
    if (
        hasattr(stereo_optimizer, "discarded_frames")
        and stereo_optimizer.discarded_frames
    ):
        print(f"Frames removed as outliers: {len(stereo_optimizer.discarded_frames)}")

    # Print relative pose
    print("\nFinal relative pose:")
    rel_pose = results["relative_pose"]
    print(
        f"Rotation (deg): {rel_pose['euler_angles_deg'][0]:.2f}, {rel_pose['euler_angles_deg'][1]:.2f}, {rel_pose['euler_angles_deg'][2]:.2f}"
    )
    tvec = np.asarray(rel_pose["tvec"]).flatten()
    print(f"Translation: [{tvec[0]:.2f}, {tvec[1]:.2f}, {tvec[2]:.2f}]")


def load_calib_data(calib_file_path):
    """Load calibration data from pickle file."""
    with open(calib_file_path, "rb") as f:
        return pickle.load(f)


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


if __name__ == "__main__":
    # result_stereo = run_cv_stereo(
    #     left_pkl_path="pycalib/data/cache/calib_result_Projector_cv.pkl",
    #     right_pkl_path="pycalib/data/cache/calib_result_Camera_cv.pkl",
    #     output_path="pycalib/results/stereo_calib_report_cv.json",
    #     left_name="Projector",
    #     right_name="Camera",
    #     img_size_left=(1280, 720),
    #     img_size_right=(1280, 720),
    # )

    run_new_stereo(
        left_calib_file="pycalib/data/cache/calib_result_Camera_cv.pkl",
        right_calib_file="pycalib/data/cache/calib_result_Projector_cv.pkl",
    )
