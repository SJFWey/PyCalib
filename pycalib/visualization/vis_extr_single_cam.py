import json
import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def visualize_extrinsics_camera_centered(
    calib_results_path,
    feature_data_path,
    figsize=(10, 8),
    elevation=-54,
    azimuth=-48,
    roll=-47,
    point_color="blue",
    point_size=5,
    camera_scale=1.0,
):
    """
    Visualize object points transformed into the camera coordinate system.
    """
    calib_results_path = Path(calib_results_path)
    if not calib_results_path.is_file():
        raise FileNotFoundError(
            f"Calibration results file not found: {calib_results_path}"
        )
    with open(calib_results_path, "r") as f:
        calib_data = json.load(f)

    cam_name = calib_data.get("cam_name", "Camera")

    extrinsics = calib_data.get("extrinsics", {})

    feature_data_path = Path(feature_data_path)
    if not feature_data_path.is_file():
        raise FileNotFoundError(f"Feature data file not found: {feature_data_path}")
    with open(feature_data_path, "rb") as f:
        feature_data = pickle.load(f)

    all_points_in_cam_frame = []
    debug_printed = False

    for frame_id in feature_data:
        frame_features = feature_data[frame_id]
        object_points_world = np.array(frame_features["objectPoints"], dtype=np.float64)
        if object_points_world.ndim != 2 or object_points_world.shape[1] != 3:
            print(
                f"Warning: Invalid objectPoints shape {object_points_world.shape} for frame {frame_id}. Skipping."
            )
            continue

        rvec = np.array(extrinsics["rvec"], dtype=np.float64).reshape(3, 1)
        tvec = np.array(extrinsics["tvec"], dtype=np.float64).reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)

        points_cam = (R @ object_points_world.T + tvec).T

        all_points_in_cam_frame.append(points_cam)

    if not all_points_in_cam_frame:
        raise ValueError("No valid points could be processed or transformed.")

    all_points_np = np.concatenate(all_points_in_cam_frame, axis=0)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    _draw_camera_origin(ax, scale=camera_scale, label=f"{cam_name} Frame")

    ax.scatter(
        all_points_np[:, 0],
        all_points_np[:, 1],
        all_points_np[:, 2],
        c=point_color,
        s=point_size,
        marker=".",
        alpha=0.6,
        label="Object Points (in Camera Frame)",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    _set_axes_equal(ax)

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    z_lim = ax.get_zlim()
    ax.set_box_aspect(
        (
            abs(x_lim[1] - x_lim[0]),
            abs(y_lim[1] - y_lim[0]),
            abs(z_lim[1] - z_lim[0]) * 1.5,
        )
    )

    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        fontsize="small",
        loc="upper right",
        framealpha=0.5,
    )

    ax.view_init(elev=elevation, azim=azimuth, roll=roll)
    plt.tight_layout()
    plt.show()


def _draw_camera_origin(ax, scale=1.0, label="Camera Frame"):
    """
    Draw a reference camera and coordinate axes at the origin (0,0,0).
    """
    cam_points = np.array(
        [
            [0, 0, 0],
            [-scale * 0.5, -scale * 0.5, scale],
            [scale * 0.5, -scale * 0.5, scale],
            [scale * 0.5, scale * 0.5, scale],
            [-scale * 0.5, scale * 0.5, scale],
        ]
    )
    pyramid_color = "red"

    ax.plot(
        [
            cam_points[1, 0],
            cam_points[2, 0],
            cam_points[3, 0],
            cam_points[4, 0],
            cam_points[1, 0],
        ],
        [
            cam_points[1, 1],
            cam_points[2, 1],
            cam_points[3, 1],
            cam_points[4, 1],
            cam_points[1, 1],
        ],
        [
            cam_points[1, 2],
            cam_points[2, 2],
            cam_points[3, 2],
            cam_points[4, 2],
            cam_points[1, 2],
        ],
        color=pyramid_color,
        linewidth=1.5,
        label=label,
    )
    for i in range(1, 5):
        ax.plot(
            [cam_points[0, 0], cam_points[i, 0]],
            [cam_points[0, 1], cam_points[i, 1]],
            [cam_points[0, 2], cam_points[i, 2]],
            color=pyramid_color,
            linewidth=1.5,
            alpha=0.8,
        )

    axis_length = scale * 1.5
    ax.plot([0, axis_length], [0, 0], [0, 0], color="red", linewidth=2, label="X")
    ax.text(axis_length * 1.1, 0, 0, "X", color="red")
    ax.plot([0, 0], [0, axis_length], [0, 0], color="green", linewidth=2, label="Y")
    ax.text(0, axis_length * 1.1, 0, "Y", color="green")
    ax.plot([0, 0], [0, 0], [0, axis_length], color="blue", linewidth=2, label="Z")
    ax.text(0, 0, axis_length * 1.1, "Z", color="blue")


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


if __name__ == "__main__":
    results_dir = Path("pycalib/results")
    data_dir = Path("pycalib/data/cache")

    cam_calib_file = results_dir / "Camera_calib_report.json"
    cam_feature_file = data_dir / "cam_feature_data.pkl"

    visualize_extrinsics_camera_centered(
        calib_results_path=cam_calib_file,
        feature_data_path=cam_feature_file,
        camera_scale=1.0,
        point_size=5,
    )

    proj_calib_file = results_dir / "Projector_calib_report.json"
    proj_feature_file = data_dir / "proj_feature_data.pkl"

    visualize_extrinsics_camera_centered(
        calib_results_path=proj_calib_file, feature_data_path=proj_feature_file
    )
