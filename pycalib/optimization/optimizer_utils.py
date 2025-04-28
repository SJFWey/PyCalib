import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from pycalib.optimization.optimizer_configs import (
    Extrinsics,
    Intrinsics,
    OptimizationState,
)


def estimate_focal(object_points, image_points, cx, cy):
    """
    Estimate camera focal lengths assuming a planar calibration target.
    Uses RANSAC to compute homographies and simple least squares to solve for intrinsic parameters.
    """
    valid_homographies = []

    principal_point_offset = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])

    for i in sorted(image_points.keys()):
        img_pts = image_points[i]
        obj_pts = object_points[i]

        if img_pts is None or obj_pts is None or np.isnan(img_pts).any():
            continue

        obj_pts_2d = np.array(obj_pts)[:, :2]
        img_pts_2d = np.array(img_pts).reshape(-1, 2)

        if obj_pts_2d.shape != img_pts_2d.shape:
            continue

        # Find homography using RANSAC
        H, mask = cv2.findHomography(obj_pts_2d, img_pts_2d, cv2.RANSAC, 3.0)

        if H is not None and not np.isnan(H).any() and not np.isinf(H).any():
            valid_homographies.append(H)

    if len(valid_homographies) < 5:
        raise ValueError(
            f"[estimate_focal] Not enough valid homographies ({len(valid_homographies)}), cannot estimate focal length"
        )
        # Returning None is handled by the exception, but kept return type hint consistent if needed later
        # return None, None # This line is unreachable due to raise

    a_list = []
    b_list = []

    for H in valid_homographies:
        h_offset = principal_point_offset @ H
        # Normalize homography
        if np.abs(h_offset[2, 2]) > 1e-8:
            h_offset = h_offset / h_offset[2, 2]
        else:
            continue  # Avoid division by zero if H(3,3) is close to zero

        h1 = h_offset[:, 0]
        h2 = h_offset[:, 1]

        # Constraints based on orthogonality of rotation matrix columns represented by homography
        a_list.append([h1[0] * h2[0], h1[1] * h2[1]])
        b_list.append(-h1[2] * h2[2])

        a_list.append([h1[0] ** 2 - h2[0] ** 2, h1[1] ** 2 - h2[1] ** 2])
        b_list.append(h2[2] ** 2 - h1[2] ** 2)

    A = np.array(a_list)
    b = np.array(b_list)

    try:
        # Solve the linear system A * [1/fx^2, 1/fy^2]' = b
        solution, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)

        inv_fx_sq = np.abs(solution[0])
        inv_fy_sq = np.abs(solution[1])

        if inv_fx_sq > 1e-8 and inv_fy_sq > 1e-8:
            fx = 1.0 / np.sqrt(inv_fx_sq)
            fy = 1.0 / np.sqrt(inv_fy_sq)
        else:
            raise ValueError(
                "[estimate_focal] Invalid focal length solution (inverse square close to zero)."
            )
            # return None, None # Unreachable

        # Check if estimated focal lengths are within a reasonable range
        if fx < 100 or fx > 5000 or fy < 100 or fy > 5000:
            raise ValueError(
                f"[estimate_focal] Estimated focal lengths out of reasonable range: fx={fx:.2f}, fy={fy:.2f}"
            )
            # return None, None # Unreachable

        return fx, fy

    except Exception as e:
        # Catch potential errors during least squares or subsequent calculations
        raise ValueError(f"[estimate_focal] Error in focal length estimation: {e}")


def normalize_pixel(image_points: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
    """
    Normalizes pixel image coordinates using intrinsic camera parameters.

    Args:
        image_points: Pixel coordinates (Nx2 or 2xN).
        intrinsics: Intrinsics object containing fx, fy, cx, cy, k, alpha_c.

    Returns:
        x_normalized: Normalized, undistorted image points (Nx2).
    """
    if image_points.shape[0] != 2 and image_points.shape[1] == 2:
        x_pixels = image_points.T
    elif image_points.shape[0] == 2:
        x_pixels = image_points
    else:
        raise ValueError("Input image_points should be Nx2 or 2xN")

    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.cx
    cy = intrinsics.cy
    dist_coeffs = intrinsics.dist_coeffs

    if fx == 0 or fy == 0:
        raise ValueError("Focal lengths (fx, fy) cannot be zero.")

    x_distorted = np.vstack([(x_pixels[0, :] - cx) / fx, (x_pixels[1, :] - cy) / fy])

    k1, k2, p1, p2, k3 = dist_coeffs

    x_normalized = x_distorted.copy()

    for _ in range(20):
        r_sq = np.sum(x_normalized**2, axis=0)
        k_radial = 1 + k1 * r_sq + k2 * r_sq**2 + k3 * r_sq**3
        delta_x = np.vstack(
            [
                2 * p1 * x_normalized[0, :] * x_normalized[1, :]
                + p2 * (r_sq + 2 * x_normalized[0, :] ** 2),
                p1 * (r_sq + 2 * x_normalized[1, :] ** 2)
                + 2 * p2 * x_normalized[0, :] * x_normalized[1, :],
            ]
        )
        x_normalized = (x_distorted - delta_x) / k_radial[np.newaxis, :]

    return x_normalized.T


def estimate_init_extrinsics(
    image_points: np.ndarray, object_points: np.ndarray, intrinsics: Intrinsics
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes an initial estimate of extrinsic parameters using Direct Linear Transform (DLT).

    Args:
        image_points: A numpy array of shape (N,2 or 2,N) containing pixel image coordinates.
        object_points: A numpy array of shape (N,3 or 3,N) containing corresponding 3D object coordinates.
        intrinsics: An Intrinsics object containing camera intrinsic parameters.

    Returns:
        Extrinsics object containing:
        - rvec: Estimated rotation vector (3x1).
        - tvec: Estimated translation vector (3x1).
        and:
        - rotation_matrix: Estimated rotation matrix (3x3).
    """
    if image_points.shape[0] != 2 and image_points.shape[1] == 2:
        image_points = image_points.T
    elif image_points.shape[0] == 2:
        image_points = image_points
    else:
        raise ValueError("Input image_points should be Nx2 or 2xN")

    if object_points.shape[0] != 3 and object_points.shape[1] == 3:
        object_points = object_points.T
    elif object_points.shape[0] == 3:
        object_points = object_points
    else:
        raise ValueError("Input object_points should be Nx3 or 3xN")

    num_points = object_points.shape[1]
    if image_points.shape[1] != num_points:
        raise ValueError("Number of image points and object points must match.")

    normalized_image_points = normalize_pixel(image_points, intrinsics)

    jacobian_matrix = np.zeros((2 * num_points, 12))

    for i in range(num_points):
        x_world = object_points[:, i]  # 3D point (3x1)
        x_norm, y_norm = normalized_image_points[i, :]  # Normalized image point

        # Row 2*i (x-coordinate equation)
        # -X.T for r11, r12, r13 columns
        jacobian_matrix[2 * i, 0:3] = -x_world.T
        # Zeros for r21, r22, r23 columns
        jacobian_matrix[2 * i, 3:6] = 0
        # x_norm * X.T for r31, r32, r33 columns
        jacobian_matrix[2 * i, 6:9] = x_norm * x_world.T
        # -1 for tx, 0 for ty, and x_norm for tz
        jacobian_matrix[2 * i, 9] = -1
        jacobian_matrix[2 * i, 10] = 0
        jacobian_matrix[2 * i, 11] = x_norm

        # Row 2*i+1 (y-coordinate equation)
        # Zeros for r11, r12, r13 columns
        jacobian_matrix[2 * i + 1, 0:3] = 0
        # -X.T for r21, r22, r23 columns
        jacobian_matrix[2 * i + 1, 3:6] = -x_world.T
        # y_norm * X.T for r31, r32, r33 columns
        jacobian_matrix[2 * i + 1, 6:9] = y_norm * x_world.T
        # 0 for tx, -1 for ty, and y_norm for tz
        jacobian_matrix[2 * i + 1, 9] = 0
        jacobian_matrix[2 * i + 1, 10] = -1
        jacobian_matrix[2 * i + 1, 11] = y_norm

    # Solve the linear system J^T * J * p = 0
    j_transpose_j = jacobian_matrix.T @ jacobian_matrix
    _, _, v_transpose = np.linalg.svd(j_transpose_j)

    # Solution is the eigenvector corresponding to the smallest eigenvalue (last row of V^T)
    solution_vector = v_transpose[-1, :]

    # Extract rotation matrix (scaled) and translation vector (scaled)
    r_scaled = solution_vector[:9].reshape(3, 3)
    t_scaled = solution_vector[9:].reshape(3, 1)

    # Find the closest valid rotation matrix using SVD
    u_rot, s_rot, v_transpose_rot = np.linalg.svd(r_scaled)
    rotation_matrix = u_rot @ v_transpose_rot

    # Check and correct for improper rotation (reflection)
    if np.linalg.det(rotation_matrix) < 0:
        # If determinant is negative, flip the sign of the last column of V^T
        v_transpose_rot[-1, :] *= -1
        rotation_matrix = u_rot @ v_transpose_rot

    # Calculate the scale factor for translation
    # Use the ratio of the Frobenius norms
    r_scaled_norm = np.linalg.norm(r_scaled, "fro")
    r_corrected_norm = np.linalg.norm(rotation_matrix, "fro")

    # Avoid division by zero
    scale = max(r_scaled_norm / r_corrected_norm, 1e-8)

    # Scale the translation vector
    tvec = t_scaled / scale

    # Convert rotation matrix to rotation vector
    rvec, _ = cv2.Rodrigues(rotation_matrix)

    return rvec, tvec, rotation_matrix


def plot_cost_history(
    optim_state: OptimizationState,
    cam_name: str,
    logger,
    save_path=None,
    show_plot=True,
):
    """Plots the optimization cost history."""
    if not optim_state.cost_history:
        logger.warning("[plot_cost_history] No cost history available to plot.")
        return

    iterations = range(len(optim_state.cost_history))
    costs = optim_state.cost_history

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, costs, marker="o", linestyle="-")
    plt.xlabel("Iteration (Function Evaluation)")
    plt.ylabel("Cost (Sum of Squared Residuals)")
    plt.title(f"Optimization Cost History for Camera: {cam_name}")
    plt.yscale("log")  # Use log scale for better visualization of large cost changes
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path)
        except Exception as e:
            logger.error(f"[plot_cost_history] Failed to save plot: {e}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def visualize_reproj_per_frame(
    cam_name: str,
    valid_frames: set,
    obj_pts_all_frames: dict,
    img_pts_all_frames: dict,
    intrinsics: Intrinsics,
    extrinsics: Extrinsics,
    image_size: tuple,
    logger,
    save_dir=None,
):
    """Visualizes the reprojection error for each frame and overall statistics."""
    if save_dir is None:
        # Default save directory if none provided
        save_dir = os.path.join(
            "pycalib", "results", "debugging", "single", f"{cam_name}"
        )

    os.makedirs(save_dir, exist_ok=True)

    all_point_error_magnitudes = []
    per_frame_rms_errors = []
    frame_indices_for_rms = []

    # Prepare camera parameters in OpenCV format
    camera_matrix = np.array(
        [
            [intrinsics.fx, 0, intrinsics.cx],
            [0, intrinsics.fy, intrinsics.cy],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    distortion_coeffs = intrinsics.dist_coeffs
    distortion_coeffs = np.asarray(distortion_coeffs, dtype=np.float64).reshape(-1)

    rotation_vector = np.asarray(extrinsics.rvec, dtype=np.float64).reshape(
        3,
    )
    translation_vector = np.asarray(extrinsics.tvec, dtype=np.float64).reshape(
        3,
    )

    for frame in sorted(valid_frames):
        if frame not in obj_pts_all_frames or frame not in img_pts_all_frames:
            logger.warning(f"Frame {frame}: Missing data, skipping visualization.")
            continue

        # Ensure points are numpy arrays with correct shape
        object_points_frame = np.asarray(
            obj_pts_all_frames[frame], dtype=np.float64
        ).reshape(-1, 3)
        image_points_frame = np.asarray(
            img_pts_all_frames[frame], dtype=np.float64
        ).reshape(-1, 2)

        if object_points_frame.shape[0] == 0 or image_points_frame.shape[0] == 0:
            logger.warning(
                f"Frame {frame}: No points available, skipping visualization."
            )
            continue

        # Project 3D points to 2D image plane
        projected_points, _ = cv2.projectPoints(
            object_points_frame,
            rotation_vector,
            translation_vector,
            camera_matrix,
            distortion_coeffs,
        )
        projected_points = projected_points.reshape(-1, 2)

        # --- Plotting for the current frame ---
        plt.figure(figsize=(10, 8))
        # Observed points
        plt.scatter(
            image_points_frame[:, 0],
            image_points_frame[:, 1],
            c="blue",
            marker=".",
            label="Observed",
            s=10,
        )
        # Reprojected points
        plt.scatter(
            projected_points[:, 0],
            projected_points[:, 1],
            c="red",
            marker="x",
            label="Projected",
            s=10,
        )
        # Draw lines connecting observed and reprojected points (error vectors)
        for j in range(len(image_points_frame)):
            plt.plot(
                [image_points_frame[j, 0], projected_points[j, 0]],
                [image_points_frame[j, 1], projected_points[j, 1]],
                "g-",
                alpha=0.3,
                linewidth=0.5,
            )

        # Calculate errors for this frame
        error_vectors = projected_points - image_points_frame
        error_magnitudes = np.linalg.norm(error_vectors, axis=1)
        all_point_error_magnitudes.extend(error_magnitudes)

        # Calculate RMS error for this frame
        rms_error = (
            np.sqrt(np.mean(np.square(error_magnitudes)))
            if error_magnitudes.size > 0
            else np.nan
        )
        if not np.isnan(rms_error):
            per_frame_rms_errors.append(rms_error)
            frame_indices_for_rms.append(frame)

        # Plot configuration
        plt.title(
            f"{cam_name} - Frame {frame} ({len(image_points_frame)} valid points): RMS={rms_error:.3f}px"
        )
        plt.xlabel("X (pixels)")
        plt.ylabel("Y (pixels)")
        plt.xlim(0, image_size[0])
        plt.ylim(image_size[1], 0)  # Origin at top-left
        plt.gca().set_aspect("equal", adjustable="box")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        # Save the plot for the current frame
        save_path = os.path.join(save_dir, f"{cam_name}_frame_{frame}.png")
        try:
            plt.savefig(save_path, dpi=150)
        except Exception as e:
            logger.error(f"Failed to save frame {frame} plot: {e}")
        plt.close()  # Close the plot figure to free memory

    # --- Plotting overall RMS error per frame ---
    if per_frame_rms_errors and frame_indices_for_rms:
        plt.figure(
            figsize=(max(10, len(frame_indices_for_rms) * 0.5), 6)
        )  # Adjust width based on number of frames
        plt.bar(
            frame_indices_for_rms, per_frame_rms_errors, edgecolor="black", width=0.6
        )
        plt.title(f"{cam_name} - Per-Frame RMS Error")
        plt.xlabel("Frame Index")
        plt.ylabel("RMS Error (pixels)")
        plt.xticks(
            frame_indices_for_rms, rotation=45, ha="right"
        )  # Rotate labels if many frames
        plt.grid(True, axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        save_path_rms_chart = os.path.join(save_dir, f"{cam_name}_per_frame_rms.png")
        try:
            plt.savefig(save_path_rms_chart, dpi=150)
            logger.info(f"Saved per-frame RMS chart to {save_path_rms_chart}")
        except Exception as e:
            logger.error(f"Failed to save per-frame RMS chart: {e}")
        plt.close()
    else:
        logger.warning(
            "No valid per-frame RMS errors calculated, skipping RMS chart plot."
        )

    # --- Plotting histogram of all point errors ---
    if all_point_error_magnitudes:
        plt.figure(figsize=(10, 6))
        # Determine number of bins, e.g., using square root rule or fixed number
        num_bins = max(50, int(np.sqrt(len(all_point_error_magnitudes))))
        plt.hist(
            all_point_error_magnitudes,
            bins=num_bins,
            edgecolor="black",
        )
        # Calculate overall statistics
        median_error = np.median(all_point_error_magnitudes)
        mean_error = np.mean(all_point_error_magnitudes)
        rms_overall = np.sqrt(np.mean(np.square(all_point_error_magnitudes)))
        # Add vertical lines for statistics
        plt.axvline(
            mean_error,
            color="r",
            linestyle="dashed",
            linewidth=1,
            label=f"Mean={mean_error:.3f}",
        )
        plt.axvline(
            median_error,
            color="g",
            linestyle="dashed",
            linewidth=1,
            label=f"Median={median_error:.3f}",
        )
        plt.axvline(
            rms_overall,
            color="purple",
            linestyle="dashed",
            linewidth=1,
            label=f"RMS={rms_overall:.3f}",
        )

        plt.title(
            f"{cam_name} - Histogram of All Point Reprojection Errors ({len(all_point_error_magnitudes)} points)"
        )
        plt.xlabel("Error Magnitude (pixels)")
        plt.ylabel("Number of Points")
        plt.yscale("log")  # Use log scale for y-axis if error distribution is skewed
        plt.legend()
        plt.grid(True, axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        save_path_all_hist = os.path.join(
            save_dir, f"{cam_name}_all_points_error_histogram.png"
        )
        try:
            plt.savefig(save_path_all_hist, dpi=150)
            logger.info(f"Saved all points error histogram to {save_path_all_hist}")
        except Exception as e:
            logger.error(f"Failed to save all points error histogram: {e}")
        plt.close()
    else:
        logger.warning(
            "No individual point errors calculated, skipping all points error histogram plot."
        )
