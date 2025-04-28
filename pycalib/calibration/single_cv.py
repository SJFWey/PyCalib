import json
import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from pycalib.feature_processing.feature_processing_utils import filter_features
from pycalib.optimization.optimizer_configs import Extrinsics, Intrinsics

color_divide = "\033[1;32;40m"
color_end = "\033[0m"


def visualize_reprojection(
    feature_data,
    calibration_result=None,
    cam_name="Unknown",
    output_dir="pycalib/results/debugging/single_cv",
):
    """
    Unified visualization function for calibration data

    Args:
        feature_data: Original feature data dictionary
        calibration_result: Calibration result dictionary (required for reprojection)
        cam_name: Name of the camera or projector
        output_dir: Base directory for outputs
    """

    if calibration_result is None:
        raise ValueError(
            "calibration_result is required for reprojection visualization"
        )

    output_dir = Path(f"{output_dir}/{cam_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_matrix = np.array(calibration_result["camera_matrix"])
    dist_coeffs = np.array(calibration_result["dist_coeffs"])

    # Determine image size based on camera name
    if "Projector" in cam_name:
        img_size = (768, 768)
    else:
        img_size = (1280, 720)

    processed_frames = 0
    for frame_info in calibration_result.get("frames", []):
        # Simplified check for valid frame info structure
        if not isinstance(frame_info, dict) or not all(
            k in frame_info for k in ["frame_idx", "rvec", "tvec"]
        ):
            continue

        frame_idx = frame_info["frame_idx"]
        rvec = np.array(frame_info.get("rvec"))
        tvec = np.array(frame_info.get("tvec"))
        frame_error = frame_info.get("error", "N/A")

        if frame_idx not in feature_data:
            continue

        frame_feature_data = feature_data[frame_idx]
        if (
            "imagePoints" not in frame_feature_data
            or "objectPoints" not in frame_feature_data
        ):
            continue

        img_points = np.array(
            frame_feature_data["imagePoints"], dtype=np.float32
        ).reshape(-1, 2)
        obj_points = np.array(frame_feature_data["objectPoints"], dtype=np.float32)

        if img_points.size == 0 or obj_points.size == 0:
            continue

        projected_points, _ = cv2.projectPoints(
            obj_points, rvec, tvec, camera_matrix, dist_coeffs
        )
        projected_points = projected_points.reshape(-1, 2)

        fig, ax = plt.subplots(figsize=(10, 10 * img_size[1] / img_size[0]))
        ax.plot(
            img_points[:, 0], img_points[:, 1], "bo", markersize=4, label="Observed"
        )
        ax.plot(
            projected_points[:, 0],
            projected_points[:, 1],
            "rx",
            markersize=4,
            label="Reprojected",
        )

        for i in range(len(img_points)):
            ax.plot(
                [img_points[i, 0], projected_points[i, 0]],
                [img_points[i, 1], projected_points[i, 1]],
                "g-",
                linewidth=0.5,
                alpha=0.6,
            )

        ax.set_xlim(0, img_size[0])
        ax.set_ylim(img_size[1], 0)
        ax.set_xlabel("Image X (pixels)")
        ax.set_ylabel("Image Y (pixels)")

        error_val_str = (
            f"{frame_error:.4f}" if isinstance(frame_error, (int, float)) else "N/A"
        )
        ax.set_title(
            f"{cam_name} - Frame {frame_idx} Reprojection\nAvg Error: {error_val_str} pixels"
        )
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()
        plt.savefig(
            output_dir / f"{cam_name}_frame_{frame_idx:02d}_reprojection.png",
            dpi=300,
        )
        plt.close(fig)
        processed_frames += 1

    if processed_frames <= 0:
        print(f"No valid frames found to generate reprojection plots for {cam_name}.")


def organize_single_calib_result(cv_calib_result, filtered_data):
    """
    Organize OpenCV single camera calibration results into format compatible with stereo calibration
    """
    camera_matrix = np.array(cv_calib_result["camera_matrix"])
    dist_coeffs = np.array(cv_calib_result["dist_coeffs"]).flatten()
    cam_name = cv_calib_result.get("cam_name", "Unknown")

    intrinsics = Intrinsics(
        fx=camera_matrix[0, 0],
        fy=camera_matrix[1, 1],
        cx=camera_matrix[0, 2],
        cy=camera_matrix[1, 2],
        dist_coeffs=dist_coeffs,
    )

    valid_frames = [frame["frame_idx"] for frame in cv_calib_result["frames"]]

    extrinsics = {}
    for frame in cv_calib_result["frames"]:
        frame_idx = frame["frame_idx"]
        extrinsics[frame_idx] = Extrinsics(
            rvec=np.array(frame["rvec"], dtype=np.float64).reshape(3, 1),
            tvec=np.array(frame["tvec"], dtype=np.float64).reshape(3, 1),
        )

    error_report = {
        "global_stats": {"rms_error_px": cv_calib_result.get("reprojection_error", 0.0)}
    }

    if "frames" in cv_calib_result:
        error_report["per_frame_stats"] = []
        error_report["frame_stats"] = {}
        for frame in cv_calib_result["frames"]:
            if "error" in frame:
                frame_stat = {
                    "frame_id": frame["frame_idx"],
                    "error_px": frame["error"],
                }
                error_report["per_frame_stats"].append(frame_stat)
                error_report["frame_stats"][frame["frame_idx"]] = {
                    "error_px": frame["error"]
                }

    feature_data_organized = {"image_points": {}, "object_points": {}}
    for frame_idx in valid_frames:
        if frame_idx in filtered_data:
            # Ensure points are numpy arrays with correct shapes
            img_pts_list = filtered_data[frame_idx].get("imagePoints", [])
            obj_pts_list = filtered_data[frame_idx].get("objectPoints", [])

            # Convert to numpy arrays, handling potential empty lists
            img_pts_np = np.array(img_pts_list, dtype=np.float64).reshape(-1, 2)
            obj_pts_np = np.array(obj_pts_list, dtype=np.float64).reshape(-1, 3)

            feature_data_organized["image_points"][frame_idx] = img_pts_np
            feature_data_organized["object_points"][frame_idx] = obj_pts_np

    organized_result = {
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "feature_data": feature_data_organized,
        "valid_frames": valid_frames,
        "error_report": error_report,
        "cam_name": cam_name,
    }

    return organized_result


def cv_single_pipeline(
    cam_feature_path="pycalib/data/cache/filtered_cam_feature_data.pkl",
    proj_feature_path="pycalib/data/cache/filtered_proj_feature_data.pkl",
    output_dir="pycalib/results/debugging/single_cv",
):
    """OpenCV single camera calibration pipeline"""

    result_cam, filtered_cam_data = cv_single(cam_feature_path, cam_name="Camera")
    organized_cam_result = organize_single_calib_result(result_cam, filtered_cam_data)
    with open("pycalib/data/cache/calib_result_Camera_cv.pkl", "wb") as f:
        pickle.dump(organized_cam_result, f)
    with open("pycalib/results/Camera_calib_report_cv.json", "w") as f:
        json.dump(result_cam, f, indent=2)
    visualize_reprojection(
        filtered_cam_data, result_cam, cam_name="Camera", output_dir=output_dir
    )

    result_proj, filtered_proj_data = cv_single(proj_feature_path, cam_name="Projector")
    organized_proj_result = organize_single_calib_result(
        result_proj, filtered_proj_data
    )
    with open("pycalib/data/cache/calib_result_Projector_cv.pkl", "wb") as f:
        pickle.dump(organized_proj_result, f)
    with open("pycalib/results/Projector_calib_report_cv.json", "w") as f:
        json.dump(result_proj, f, indent=2)
    visualize_reprojection(
        filtered_proj_data,
        result_proj,
        cam_name="Projector",
        output_dir=output_dir,
    )


def cv_single(feature_path, cam_name="Camera"):
    """
    Performs single camera calibration using OpenCV.

    Args:
        feature_path (str or Path): Path to the feature data file (.pkl).
        cam_name (str): Name of the camera/projector.

    Returns:
        tuple: A tuple containing:
            - calibration_result (dict): Dictionary with calibration results (mtx, dist, rvecs, tvecs, error).
            - filtered_data (dict): Dictionary with the filtered feature points used for calibration.
        Returns (None, None) if calibration fails or no valid points are found.
    """
    feature_path = Path(feature_path)
    if not feature_path.exists():
        print(f"Error: Feature data file not found at {feature_path}")
        return None, None

    try:
        with open(feature_path, "rb") as infile:
            feature_data = pickle.load(infile)
    except Exception as e:
        print(f"Error loading feature data from {feature_path}: {e}")
        return None, None

    all_image_points = []
    all_object_points = []
    filtered_data = {}
    valid_frame_indices = []

    sorted_frame_indices = sorted(feature_data.keys())

    for frame_idx in sorted_frame_indices:
        data = feature_data[frame_idx]
        if "imagePoints" in data and "objectPoints" in data:
            img_points = np.array(data["imagePoints"], dtype=np.float32)
            obj_points = np.array(data["objectPoints"], dtype=np.float32)

            filtered_img_points, filtered_obj_points = filter_features(
                cam_name, frame_idx, img_points, obj_points
            )

            # Store filtered points using the original frame_idx, even if empty after filtering
            filtered_data[frame_idx] = {
                "imagePoints": filtered_img_points.tolist(),
                "objectPoints": filtered_obj_points.tolist(),
            }

            if (
                len(filtered_img_points) > 3 and len(filtered_obj_points) > 3
            ):  # Need at least 4 points
                all_image_points.append(filtered_img_points)
                all_object_points.append(filtered_obj_points)
                valid_frame_indices.append(frame_idx)
            else:
                print(
                    f"[{cam_name}] Frame {frame_idx}: Insufficient points after filtering ({len(filtered_img_points)}), skipping."
                )

    if not all_image_points or not all_object_points:
        print(
            f"Error: No valid frames with sufficient points found for {cam_name} after filtering."
        )
        # Return filtered_data even if calibration fails, for potential debugging
        return None, filtered_data

    all_image_points = [points.reshape(-1, 1, 2) for points in all_image_points]
    all_object_points = [points.reshape(-1, 1, 3) for points in all_object_points]

    dist_initial = np.zeros((5, 1))  # Initialize distortion coefficients guess

    if cam_name == "Camera" or cam_name == "camera":
        K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
        img_size = (1280, 720)
        flags = (
            cv2.CALIB_USE_INTRINSIC_GUESS
            | cv2.CALIB_FIX_PRINCIPAL_POINT
            | cv2.CALIB_ZERO_TANGENT_DIST
            | cv2.CALIB_FIX_K3
        )
    elif cam_name == "Projector" or cam_name == "projector":
        K = np.array([[1300.0, 0, 384], [0, 1300.0, 384], [0, 0, 1]])
        img_size = (768, 768)
        flags = (
            cv2.CALIB_USE_INTRINSIC_GUESS
            # | cv2.CALIB_FIX_PRINCIPAL_POINT
            | cv2.CALIB_ZERO_TANGENT_DIST
            | cv2.CALIB_FIX_K2
            | cv2.CALIB_FIX_K3
        )
    else:
        raise ValueError(f"Invalid camera name: {cam_name}")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_object_points,
        all_image_points,
        img_size,
        K,
        dist_initial,
        flags=flags,
    )

    # Calculate per-frame reprojection errors
    per_frame_errors = []
    for i in range(len(all_object_points)):
        imgpoints2, _ = cv2.projectPoints(
            all_object_points[i], rvecs[i], tvecs[i], mtx, dist
        )
        # Use np.float64 for error calculation to avoid precision issues
        error = cv2.norm(
            all_image_points[i].astype(np.float64).reshape(-1, 2),
            imgpoints2.astype(np.float64).reshape(-1, 2),
            cv2.NORM_L2,
        ) / len(imgpoints2)
        per_frame_errors.append(error)

    print(
        f"\n{color_divide}[{cam_name}] OpenCV RMS Reprojection error: {ret}{color_end}"
    )

    calibration_result = {
        "cam_name": cam_name,
        "camera_matrix": mtx.tolist(),
        "dist_coeffs": dist.flatten().tolist(),
        "reprojection_error": float(ret),  # Use the RMS error from OpenCV
        "frames": [],
        "image_size": img_size,
        "num_frames_used": len(valid_frame_indices),
    }

    for i, frame_idx in enumerate(valid_frame_indices):
        frame_data = {
            "frame_idx": frame_idx,
            "rvec": rvecs[i].tolist(),
            "tvec": tvecs[i].tolist(),
            "error": per_frame_errors[i],
            "num_points": len(
                all_object_points[i]
            ),  # Add number of points used in this frame
        }
        calibration_result["frames"].append(frame_data)

    # Sort frames in the result by frame_idx for consistency
    calibration_result["frames"].sort(key=lambda x: x["frame_idx"])

    return calibration_result, filtered_data


if __name__ == "__main__":
    cv_single_pipeline()
