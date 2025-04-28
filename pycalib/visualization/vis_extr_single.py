import json
import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def visualize_camera_extrinsics(
    calib_results_path,
    feature_data_path=None,
    figsize=(10, 8),
    frame_ids=None,
    elevation=-54,
    azimuth=-48,
    roll=-47,
):
    """
    Visualize camera extrinsic calibration parameters in 3D with reference plane.
    """
    # Load calibration results
    calib_results_path = Path(calib_results_path)
    with open(calib_results_path, "r") as f:
        calib_data = json.load(f)

    # Extract camera parameters
    cam_name = calib_data.get("cam_name", "Camera")

    # Check if extrinsics exist directly or need to be extracted from frames
    extrinsics = calib_data.get("extrinsics", {})

    # If no direct extrinsics field, extract from frames
    if not extrinsics and "frames" in calib_data:
        extrinsics = {}
        for frame in calib_data["frames"]:
            frame_id = str(frame.get("frame_idx", len(extrinsics)))
            extrinsics[frame_id] = {
                "rvec": frame.get("rvec", [0, 0, 0]),
                "tvec": frame.get("tvec", [0, 0, 0]),
            }

    # Check if we have valid data
    if not extrinsics:
        raise ValueError("No extrinsic calibration data found in the JSON file")

    # Filter frames if frame_ids is provided
    if frame_ids is not None:
        frame_ids = [str(fid) for fid in frame_ids]  # Convert to strings
        extrinsics = {k: v for k, v in extrinsics.items() if k in frame_ids}
        if not extrinsics:
            raise ValueError(
                f"None of the specified frame_ids {frame_ids} found in calibration data"
            )

    # Load feature data if path is provided
    feature_data = None
    if feature_data_path:
        feature_data_path = Path(feature_data_path)
        if feature_data_path.is_file():
            try:
                with open(feature_data_path, "rb") as f:
                    feature_data = pickle.load(f)
            except Exception as e:
                print(
                    f"Warning: Could not load feature data from {feature_data_path}: {e}"
                )
                feature_data = None
        else:
            print(f"Warning: Feature data file not found at {feature_data_path}")
            feature_data = None

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.colormaps["tab20c"]
    colors = cmap.colors

    # Draw reference plane from the first frame's object points
    reference_points = None
    if feature_data:
        # Get first frame id if available
        first_frame_id = next(iter(feature_data.keys()), None)

        if first_frame_id is not None:
            # Check if the first frame's data is a dictionary (expected format)
            if (
                isinstance(feature_data.get(first_frame_id), dict)
                and "objectPoints" in feature_data[first_frame_id]
            ):
                try:
                    # Extract object points, ensure they are numpy array
                    obj_points_raw = feature_data[first_frame_id]["objectPoints"]
                    if isinstance(obj_points_raw, list):
                        # Convert list of lists/tuples to numpy array
                        reference_points = np.array(obj_points_raw, dtype=float)
                    elif isinstance(obj_points_raw, np.ndarray):
                        reference_points = obj_points_raw.astype(float)
                    else:
                        print(
                            f"Warning: Unexpected type for objectPoints: {type(obj_points_raw)}. Skipping reference plane."
                        )
                        reference_points = None

                    # Check shape and content after conversion
                    if (
                        reference_points is not None
                        and reference_points.ndim == 2
                        and reference_points.shape[1] == 3
                        and reference_points.size > 0
                    ):
                        # Draw the reference plane
                        ax.scatter(
                            reference_points[:, 0],
                            reference_points[:, 1],
                            reference_points[:, 2],
                            color="gray",
                            s=20,
                            alpha=0.8,
                            marker="o",
                            label="Reference Points",
                        )

                        # Highlight the origin point (0,0,0)
                        origin_idx = np.argmin(np.sum(reference_points**2, axis=1))
                        ax.scatter(
                            reference_points[origin_idx, 0],
                            reference_points[origin_idx, 1],
                            reference_points[origin_idx, 2],
                            color="red",
                            s=100,
                            marker="x",
                            label="Origin (0,0,0)",
                        )
                    else:
                        if (
                            reference_points is not None
                        ):  # Only print warning if it wasn't already None
                            print(
                                f"Warning: Invalid reference points shape or size ({reference_points.shape if reference_points is not None else 'None'}). Skipping plane drawing."
                            )
                        reference_points = None  # Ensure it's None if invalid

                except Exception as e:
                    print(f"Error processing objectPoints: {e}")
                    reference_points = None
            else:
                print(
                    f"Warning: 'objectPoints' not found or invalid format in feature data for frame {first_frame_id}. Skipping reference plane."
                )
                reference_points = None
        else:
            print(
                "Warning: No frame ID found in feature data. Skipping reference plane."
            )
            reference_points = None

    if reference_points is None and feature_data_path:  # Only print if path was given
        print(
            "Debugging: No valid reference points found or processed from feature data. Skipping plane drawing."
        )

    # Draw camera for each frame relative to the world coordinate system
    for i, frame_id in enumerate(extrinsics):
        frame_data = extrinsics[frame_id]

        # Extract rotation vector and translation vector
        rvec = np.array(frame_data.get("rvec", [0, 0, 0]))
        tvec = np.array(frame_data.get("tvec", [0, 0, 0]))

        # Draw camera for this frame
        color_idx = i % len(colors)
        _draw_camera(ax, rvec, tvec, colors[color_idx], scale=0.5, label=frame_id)

    # Set axis labels and title
    ax.set_xlabel("X (world frame)")
    ax.set_ylabel("Y (world frame)")
    ax.set_zlabel("Z (world frame)")
    ax.set_title(f"{cam_name} Extrinsic Calibration")

    # Set equal aspect ratio
    _set_axes_equal(ax)

    # --- Add this block to adjust Z limits ---
    z_limits = ax.get_zlim()
    # Ensure Z=0 is visible, add a small positive margin if max is close to 0
    new_z_max = max(z_limits[1], 0.5)  # Ensure the upper limit is at least 0.5
    ax.set_zlim(z_limits[0], new_z_max)

    # Add grid and small legend
    ax.grid(True)
    ax.legend(fontsize="small", loc="upper right", framealpha=0.5)

    # Set camera perspective - adjusted to flip the view
    ax.view_init(elev=elevation, azim=azimuth, roll=roll)
    plt.tight_layout()
    plt.show()
    # Removed return fig, ax as the plot is now shown directly


def _draw_camera(ax, rvec, tvec, color, scale=0.5, label=None):
    """
    Draw a camera at the specified pose in world coordinates.
    """
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(np.array(rvec).flatten())

    # Camera center in world coordinates
    C = -R.T @ tvec.reshape(3, 1)

    # Define a simple camera shape (pyramid)
    cam_points = np.array(
        [
            [0, 0, 0],  # Apex (camera center)
            [-scale, -scale, scale * 2],  # Front bottom left
            [scale, -scale, scale * 2],  # Front bottom right
            [scale, scale, scale * 2],  # Front top right
            [-scale, scale, scale * 2],  # Front top left
        ]
    )

    # Transform camera points to world coordinate system
    cam_points_world = np.zeros_like(cam_points)
    for i, pt in enumerate(cam_points):
        cam_points_world[i] = C.flatten() + R.T @ pt

    # Draw camera as a pyramid
    # Base of the pyramid
    ax.plot(
        [
            cam_points_world[1, 0],
            cam_points_world[2, 0],
            cam_points_world[3, 0],
            cam_points_world[4, 0],
            cam_points_world[1, 0],
        ],
        [
            cam_points_world[1, 1],
            cam_points_world[2, 1],
            cam_points_world[3, 1],
            cam_points_world[4, 1],
            cam_points_world[1, 1],
        ],
        [
            cam_points_world[1, 2],
            cam_points_world[2, 2],
            cam_points_world[3, 2],
            cam_points_world[4, 2],
            cam_points_world[1, 2],
        ],
        color=color,
        linewidth=1.0,
        label=label,
    )

    # Lines from apex to base corners
    for i in range(1, 5):
        ax.plot(
            [cam_points_world[0, 0], cam_points_world[i, 0]],
            [cam_points_world[0, 1], cam_points_world[i, 1]],
            [cam_points_world[0, 2], cam_points_world[i, 2]],
            color=color,
            linewidth=1.0,
            alpha=0.8,
        )


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

    ax.set_xlim3d([x_middle - radius, x_middle + radius])
    ax.set_ylim3d([y_middle - radius, y_middle + radius])
    ax.set_zlim3d([z_middle - radius, z_middle + radius])


if __name__ == "__main__":
    if_cv = False
    if if_cv:
        cam_data = "pycalib/results/Camera_calib_report_cv.json"
        proj_data = "pycalib/results/Projector_calib_report_cv.json"
    else:
        cam_data = "pycalib/results/Camera_calib_report.json"
        proj_data = "pycalib/results/Projector_calib_report.json"

    cam_feature_path = "pycalib/data/cache/cam_feature_data.pkl"
    proj_feature_path = "pycalib/data/cache/proj_feature_data.pkl"

    # Check if files exist before calling
    if not Path(cam_feature_path).is_file():
        print(f"Camera feature data file not found: {cam_feature_path}")
        cam_feature_path = None
    if not Path(proj_feature_path).is_file():
        print(f"Projector feature data file not found: {proj_feature_path}")
        proj_feature_path = None

    # Updated calls to pass the path directly
    visualize_camera_extrinsics(cam_data, cam_feature_path)
    visualize_camera_extrinsics(proj_data, proj_feature_path)
