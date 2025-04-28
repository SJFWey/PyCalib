import json
import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def _set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    radius = 0.5 * max([x_range, y_range, z_range])
    plot_radius = max(radius, 1e-4)

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    ax.set_aspect("equal", adjustable="box")


def visualize_stereo_extrinsics(
    calib_results_path,
    feature_data_path,
    figsize=(10, 8),
    elevation=20,
    azimuth=-60,
    roll=0,
    camera_scale=0.5,
    axis_scale=1.0,
    point_color="green",
    point_size=5,
):
    calib_results_path = Path(calib_results_path)
    if not calib_results_path.is_file():
        raise FileNotFoundError(
            f"Calibration results file not found: {calib_results_path}"
        )
    with open(calib_results_path, "r") as f:
        calib_data = json.load(f)

    left_cam_data = calib_data.get("left_camera", {})
    right_cam_data = calib_data.get("right_camera", {})
    relative_pose_data = calib_data.get("relative_pose", {})

    if not left_cam_data or not right_cam_data or not relative_pose_data:
        raise ValueError(
            "Invalid or incomplete stereo calibration data in the JSON file."
            " Missing 'left_camera', 'right_camera', or 'relative_pose'."
        )

    feature_data_path = Path(feature_data_path)
    if not feature_data_path.is_file():
        raise FileNotFoundError(f"Feature data file not found: {feature_data_path}")
    try:
        with open(feature_data_path, "rb") as f:
            feature_data = pickle.load(f)
    except Exception as e:
        raise IOError(f"Error loading feature data from {feature_data_path}: {e}")

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    left_extrinsics = left_cam_data.get("extrinsics", {})
    if "rvec" not in left_extrinsics or "tvec" not in left_extrinsics:
        raise ValueError("Missing or incomplete extrinsics for the left camera.")
    left_rvec = np.array(left_extrinsics["rvec"]).flatten()
    left_tvec = np.array(left_extrinsics["tvec"]).flatten()

    right_extrinsics = right_cam_data.get("extrinsics", {})
    if "rvec" not in right_extrinsics or "tvec" not in right_extrinsics:
        raise ValueError("Missing or incomplete extrinsics for the right camera.")
    right_rvec = np.array(right_extrinsics["rvec"]).flatten()
    right_tvec = np.array(right_extrinsics["tvec"]).flatten()

    all_object_points_world = []
    for frame_id in feature_data:
        frame_features = feature_data[frame_id]
        if "objectPoints" in frame_features:
            obj_pts = np.array(frame_features["objectPoints"], dtype=np.float64)
            if obj_pts.ndim == 2 and obj_pts.shape[1] == 3:
                all_object_points_world.append(obj_pts)
            else:
                print(
                    f"Warning: Invalid objectPoints shape {obj_pts.shape} for frame {frame_id}. Skipping."
                )
        else:
            print(f"Warning: Missing 'objectPoints' in frame {frame_id}. Skipping.")

    if not all_object_points_world:
        print(
            "Warning: No valid object points found in the feature data. Skipping point cloud visualization."
        )
        all_points_np = np.empty((0, 3))
    else:
        all_points_np = np.concatenate(all_object_points_world, axis=0)

    left_label = "Projector"
    right_label = "Camera"
    _draw_camera(ax, left_rvec, left_tvec, "red", scale=camera_scale, label=left_label)
    _draw_camera(
        ax, right_rvec, right_tvec, "green", scale=camera_scale, label=right_label
    )

    R_left, _ = cv2.Rodrigues(left_rvec)
    R_right, _ = cv2.Rodrigues(right_rvec)
    left_center = -R_left.T @ left_tvec
    right_center = -R_right.T @ right_tvec

    ax.plot(
        [left_center[0], right_center[0]],
        [left_center[1], right_center[1]],
        [left_center[2], right_center[2]],
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.6,
        label="Baseline",
    )

    if all_points_np.size > 0:
        ax.scatter(
            all_points_np[:, 0],
            all_points_np[:, 1],
            all_points_np[:, 2],
            c=point_color,
            s=point_size,
            marker=".",
            alpha=0.6,
            label="Object Points (World Coords)",
        )

    ax.set_xlabel("World X")
    ax.set_ylabel("World Y")
    ax.set_zlabel("World Z")
    ax.set_title("Stereo Camera Extrinsics and Object Points")

    min_coords = all_points_np.min(axis=0) if all_points_np.size > 0 else np.zeros(3)
    max_coords = all_points_np.max(axis=0) if all_points_np.size > 0 else np.zeros(3)

    all_relevant_points = np.vstack(
        (np.zeros(3), left_center, right_center, min_coords, max_coords)
    )
    center_of_mass = np.mean(all_relevant_points, axis=0)
    max_dist_from_center = np.max(
        np.linalg.norm(all_relevant_points - center_of_mass, axis=1)
    )

    plot_radius = max(max_dist_from_center * 1.1, axis_scale * 1.2)

    ax.set_xlim(center_of_mass[0] - plot_radius, center_of_mass[0] + plot_radius)
    ax.set_ylim(center_of_mass[1] - plot_radius, center_of_mass[1] + plot_radius)
    ax.set_zlim(center_of_mass[2] - plot_radius, center_of_mass[2] + plot_radius)

    _set_axes_equal(ax)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        fontsize="small",
        loc="upper right",
        framealpha=0.5,
    )
    ax.grid(True)

    ax.view_init(elev=elevation, azim=azimuth)

    plt.tight_layout()
    plt.show()

    return fig, ax


def _draw_camera(ax, rvec, tvec, color, scale=0.5, label=None, alpha=1.0):
    try:
        R, _ = cv2.Rodrigues(np.array(rvec).flatten())
        cam_center_world = -R.T @ np.array(tvec).flatten()
        R_world = R.T
    except (cv2.error, ValueError) as e:
        print(
            f"Warning: Rodrigues failed or invalid input for rvec={rvec}, tvec={tvec}. Skipping drawing camera. Error: {e}"
        )
        return

    s = scale
    cam_points_cam_frame = np.array(
        [
            [0, 0, 0],
            [-s * 0.5, -s * 0.5, s],
            [s * 0.5, -s * 0.5, s],
            [s * 0.5, s * 0.5, s],
            [-s * 0.5, s * 0.5, s],
        ]
    )

    cam_points_world = np.zeros_like(cam_points_cam_frame)
    for i, pt_cam in enumerate(cam_points_cam_frame):
        cam_points_world[i] = cam_center_world + R_world @ pt_cam

    ax.plot(
        [cam_points_world[j, 0] for j in [1, 2, 3, 4, 1]],
        [cam_points_world[j, 1] for j in [1, 2, 3, 4, 1]],
        [cam_points_world[j, 2] for j in [1, 2, 3, 4, 1]],
        color=color,
        linewidth=1.5,
        alpha=alpha,
    )

    for i in range(1, 5):
        ax.plot(
            [cam_points_world[0, 0], cam_points_world[i, 0]],
            [cam_points_world[0, 1], cam_points_world[i, 1]],
            [cam_points_world[0, 2], cam_points_world[i, 2]],
            color=color,
            linewidth=1.5,
            alpha=alpha * 0.8,
        )

    if label:
        ax.text(
            cam_center_world[0],
            cam_center_world[1],
            cam_center_world[2] - scale * 1.2,
            label,
            color=color,
            fontsize=8,
            ha="center",
            va="bottom",
            alpha=alpha,
        )


if __name__ == "__main__":
    results_dir = Path("pycalib/results")
    stereo_data_file = results_dir / "stereo_calib_report.json"
    data_dir = Path("pycalib/data/cache")
    feature_data_file = data_dir / "filtered_cam_feature_data.pkl"

    visualize_stereo_extrinsics(
        stereo_data_file,
        feature_data_path=feature_data_file,
        camera_scale=1.0,
        axis_scale=2.0,
        point_size=3,
        point_color="blue",
    )
