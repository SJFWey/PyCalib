import math
import threading
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import disk

# Thread-local storage for thread safety
thread_local = threading.local()


def polyfit(
    projector_map,
    features,
    radius=100,
    dense_thresh=0.5,
    parallel=True,
    max_workers=None,
):
    """
    Two dimensional polynomial fitting by least squares.
    Fits the functional form f(x,y) = z.

    Run in parallel mode (processing both x and y directions simultaneously)
    or sequential mode.

    Parameters
    ----------
    projector_map: np.ndarray
        Map to be fitted
    features: np.ndarray
        Feature points to use for fitting
    radius: int, default is 100
        Radius around each feature to use for fitting
    dense_thresh: float, default is 0.5
        Threshold for determining if enough points are available
    parallel: bool, default is True
        Whether to process directions in parallel
    max_workers: int, optional
        Maximum number of worker threads to use, defaults to None (uses CPU count)

    Returns
    -------
    projector_map: np.ndarray
        Fitted projector map
    feature_mask: np.ndarray
        Mask indicating which features were used
    """
    # Make a copy of projector_map to avoid race conditions
    projector_map_copy = projector_map.copy()
    feature_count = features.shape[0]
    feature_mask = np.ones(feature_count, dtype=bool)

    # Setup for thread-local storage
    if parallel:
        import concurrent.futures
        import threading

        thread_local = threading.local()

    def _process_direction(dir):
        """Process a single direction (x or y coordinate in projector map)"""
        dir_mask = np.ones(feature_count, dtype=bool)
        height, width = projector_map_copy.shape[:2]

        # Pre-compute grid coordinates
        if parallel:
            if not hasattr(thread_local, "grid_coords"):
                thread_local.y_grid, thread_local.x_grid = np.mgrid[0:height, 0:width]
            y_grid, x_grid = thread_local.y_grid, thread_local.x_grid
        else:
            y_grid, x_grid = np.mgrid[0:height, 0:width]

        # Pre-compute coefficient pairs
        kx = 2
        ky = 2
        order = 2
        coeff_indices = [
            (j, i)
            for i, j in np.ndindex((kx + 1, ky + 1))
            if order is None or i + j <= order
        ]

        for feature_idx in range(feature_count):
            # Efficient distance calculation using ROI approach
            y_center, x_center = features[feature_idx, 0], features[feature_idx, 1]

            # Calculate region of interest bounds
            y_min = max(0, int(y_center - radius))
            y_max = min(height, int(y_center + radius + 1))
            x_min = max(0, int(x_center - radius))
            x_max = min(width, int(x_center + radius + 1))

            # Extract ROI coordinates and calculate distances only within ROI
            y_roi = y_grid[y_min:y_max, x_min:x_max]
            x_roi = x_grid[y_min:y_max, x_min:x_max]

            # Vectorized distance calculation in ROI only
            radii = np.sqrt((y_roi - y_center) ** 2 + (x_roi - x_center) ** 2)

            # Get points within radius
            inds_r = radii <= radius
            y_inds, x_inds = np.where(inds_r)

            # Convert ROI indices back to original image coordinates
            y0 = y_inds + y_min
            x0 = x_inds + x_min

            # Get values at those points
            z = projector_map_copy[y0, x0, dir]

            # Filter out invalid values
            valid_mask = z > 0.0
            if not np.any(valid_mask):
                dir_mask[feature_idx] = 0
                continue

            valid_y = y0[valid_mask]
            valid_x = x0[valid_mask]
            valid_z = z[valid_mask]

            # Check if we have enough valid points
            if len(valid_z) / len(z) <= dense_thresh:
                dir_mask[feature_idx] = 0
                continue

            # Build coefficient matrix efficiently
            num_terms = len(coeff_indices)
            num_points = len(valid_z)
            a = np.zeros((num_points, num_terms))

            # Pre-compute all powers of x and y at once
            x_powers = np.zeros((kx + 1, num_points))
            y_powers = np.zeros((ky + 1, num_points))

            x_powers[0, :] = 1
            y_powers[0, :] = 1

            for p in range(1, kx + 1):
                x_powers[p] = valid_x**p

            for p in range(1, ky + 1):
                y_powers[p] = valid_y**p

            # Efficiently build design matrix
            for idx, (j, i) in enumerate(coeff_indices):
                a[:, idx] = y_powers[j] * x_powers[i]

            # Solve least squares problem
            try:
                # Use fast solve method for overdetermined system
                c, residuals, rank, s = np.linalg.lstsq(a, valid_z, rcond=None)

                # Apply fit to all points within radius
                # Pre-compute powers for prediction
                x0_powers = np.zeros((kx + 1, len(x0)))
                y0_powers = np.zeros((ky + 1, len(y0)))

                x0_powers[0, :] = 1
                y0_powers[0, :] = 1

                for p in range(1, kx + 1):
                    x0_powers[p] = x0**p

                for p in range(1, ky + 1):
                    y0_powers[p] = y0**p

                # Efficiently compute polynomial prediction
                Z = np.zeros(len(y0))
                for idx, (j, i) in enumerate(coeff_indices):
                    Z += c[idx] * y0_powers[j] * x0_powers[i]

                # Update projector_map_copy
                projector_map_copy[y0, x0, dir] = Z
            except np.linalg.LinAlgError:
                # Handle singular matrix or other linear algebra errors
                dir_mask[feature_idx] = 0

        return dir_mask

    # Process both directions (x and y)
    if parallel:
        # Execute both directions in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            dir_masks = list(executor.map(_process_direction, [0, 1]))
        # Combine masks from both directions
        feature_mask = dir_masks[0] & dir_masks[1]
    else:
        # Process sequentially
        mask_0 = _process_direction(0)
        mask_1 = _process_direction(1)
        feature_mask = mask_0 & mask_1

    # Update mask for valid projector map values
    mask_x = projector_map_copy[:, :, 0] > 1.0
    mask_y = projector_map_copy[:, :, 1] > 1.0
    mask = mask_x * mask_y
    projector_map_copy[:, :, 2] = mask

    return projector_map_copy, feature_mask


def adaptive_feature_extractor(
    img, min_area=250, max_area=3000, circularity_threshold=0.75
):
    """
    Extract feature points from a grayscale image.

    The function scales, blurs, applies CLAHE, and uses adaptive thresholding along with morphological
    operations to enhance features. It then finds contours, computes centroids, and filters them based
    on area and circularity.

    Parameters:
        img (numpy.ndarray): Grayscale input image with pixel values in [0, 1] or [0, 255].
        min_area (int): Minimum contour area (default 250).
        max_area (int): Maximum contour area (default 3000).
        circularity_threshold (float): Minimum circularity value (default 0.75).

    Returns:
        numpy.ndarray: Array of detected feature points, each as [cx, cy].
    """

    if np.max(img) <= 1.0:
        img *= 255
    img = img.astype(np.uint8)

    img = cv2.GaussianBlur(img, (3, 3), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)

    # _, thresh = cv2.threshold(
    #     img_clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    # )

    # img_blurred = cv2.GaussianBlur(img_clahe, (9, 9), 0)
    img_blurred = cv2.medianBlur(img_clahe, 9)

    img_binary = cv2.adaptiveThreshold(
        img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 37, 7
    )

    ks = 9
    kernel = np.ones((ks, ks), dtype=np.uint8)
    rr, cc = disk((int(ks / 2), int(ks / 2)), int(ks / 2))
    kernel[rr, cc] = 1

    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel, iterations=1)

    contours = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    imgPoints = []

    for contour in contours:
        m = cv2.moments(contour)
        if m["m00"] == 0:  # avoid division by zero
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        circularity = 0
        if perimeter > 0:
            circularity = 4 * math.pi * area / (perimeter**2)

        if area > min_area and area < max_area and circularity > circularity_threshold:
            imgPoints.append([cx, cy])

    imgPoints = np.asarray(imgPoints)

    # _, axs = plt.subplots(1, 2, figsize=(10, 5))
    # axs[0].imshow(img_binary, cmap="gray")
    # axs[0].set_title("Threshold Image")

    # axs[1].imshow(img, cmap="gray")
    # axs[1].scatter(imgPoints[:, 0], imgPoints[:, 1], c="red", s=7)
    # axs[1].set_title("Detected Features")

    # plt.tight_layout()
    # plt.show()
    # plt.close()

    return np.asarray(imgPoints)


def reorganize_and_align_feature_data(cam_feature_data, proj_feature_data):
    """
    Reorganize both camera and projector feature data and ensure they share identical object points.

    Args:
        cam_feature_data: Camera feature data dictionary (dict[int, dict[str, np.ndarray]])
        proj_feature_data: Projector feature data dictionary (dict[int, dict[str, np.ndarray]])

    Returns:
        Tuple of reorganized camera and projector feature data with common frames and
        matching object points, maintaining the original structure.
    """
    # Find frames that exist in both datasets
    common_frames = set(cam_feature_data.keys()) & set(proj_feature_data.keys())
    valid_frames = []

    for idx in sorted(common_frames):
        # Check if frame has valid camera data
        if (
            idx in cam_feature_data
            and "imagePoints" in cam_feature_data[idx]
            and "objectPoints" in cam_feature_data[idx]
            and isinstance(cam_feature_data[idx]["imagePoints"], np.ndarray)
            and isinstance(cam_feature_data[idx]["objectPoints"], np.ndarray)
            and cam_feature_data[idx]["imagePoints"].size > 0
            and cam_feature_data[idx]["objectPoints"].size > 0
        ):
            # Check if frame has valid projector data
            if (
                idx in proj_feature_data
                and "imagePoints" in proj_feature_data[idx]
                and "objectPoints" in proj_feature_data[idx]
                and isinstance(proj_feature_data[idx]["imagePoints"], np.ndarray)
                and isinstance(proj_feature_data[idx]["objectPoints"], np.ndarray)
                and proj_feature_data[idx]["imagePoints"].size > 0
                and proj_feature_data[idx]["objectPoints"].size > 0
            ):
                valid_frames.append(idx)

    # Initialize reorganized data structures (maintaining original structure)
    reorganized_cam = {}
    reorganized_proj = {}

    for frame_idx in valid_frames:
        # Convert object points to hashable format for comparison
        cam_obj_points = cam_feature_data[frame_idx]["objectPoints"]
        proj_obj_points = proj_feature_data[frame_idx]["objectPoints"]

        # Find common object points using set operations
        # Convert arrays to tuples of tuples for hashability
        cam_obj_tuples = [tuple(map(float, point)) for point in cam_obj_points]
        proj_obj_tuples = [tuple(map(float, point)) for point in proj_obj_points]

        # Find common object points
        common_obj_tuples = set(cam_obj_tuples) & set(proj_obj_tuples)

        if not common_obj_tuples:
            # Skip this frame if no common object points
            continue

        # Convert back to array
        common_obj_points = np.array(list(common_obj_tuples), dtype=np.float32)

        # Create masks for filtering image points
        cam_mask = np.array(
            [tuple(point) in common_obj_tuples for point in cam_obj_tuples]
        )
        proj_mask = np.array(
            [tuple(point) in common_obj_tuples for point in proj_obj_tuples]
        )

        # Filter image points to match common object points
        cam_img_points = cam_feature_data[frame_idx]["imagePoints"][cam_mask]
        proj_img_points = proj_feature_data[frame_idx]["imagePoints"][proj_mask]

        # Ensure dimensions match
        assert len(cam_img_points) == len(common_obj_points), (
            f"Frame {frame_idx}: Camera image points don't match object points"
        )
        assert len(proj_img_points) == len(common_obj_points), (
            f"Frame {frame_idx}: Projector image points don't match object points"
        )

        # Store the aligned data (maintaining original structure)
        reorganized_cam[frame_idx] = {
            "imagePoints": cam_img_points,
            "objectPoints": common_obj_points,
        }

        reorganized_proj[frame_idx] = {
            "imagePoints": proj_img_points,
            "objectPoints": common_obj_points,
        }

        # Copy any additional data from camera features
        for key in cam_feature_data[frame_idx]:
            if key not in ["imagePoints", "objectPoints"]:
                reorganized_cam[frame_idx][key] = cam_feature_data[frame_idx][key]

    return reorganized_cam, reorganized_proj


def filter_features(
    cam_name,
    folder_idx,
    img_points,
    obj_points,
    radius_factor=1.0,
):
    """
    Filter feature points based on spatial proximity and grid structure consistency.

    Applies RANSAC homography fitting and iterative radius-based outlier removal.
    Generates a visualization of the filtering stages.

    Parameters
    ----------
    cam_name : str
        Name of the camera or context for saving plots.
    folder_idx : int or str
        Index or identifier of the folder/frame being processed.
    img_points : ndarray
        Nx2 array of image points (u, v) to filter.
    obj_points : ndarray
        Nx3 array of corresponding object points (X, Y, Z).
    radius_factor : float, default is 1.0
        Factor (0.0-1.0) for radius-based filtering. A smaller value means stricter filtering.
        A value of 1.0 effectively disables radius filtering.

    Returns
    -------
    tuple
        Filtered image points (ndarray) and corresponding object points (ndarray).
    """
    # --- Hardcoded Filter Parameters ---
    use_grid_filter = True
    ransac_thresh = 5.0
    visualize_filtering = True  # Control whether to generate plots
    # ---

    # Essential Input validation
    if img_points is None or obj_points is None:
        print(
            f"[{cam_name}-Frame{folder_idx}] Warning: Null points provided for filtering."
        )
        return np.array([]), np.array([])

    img_points = np.asarray(img_points, dtype=np.float32)
    obj_points = np.asarray(obj_points, dtype=np.float32)

    if img_points.ndim != 2 or img_points.shape[1] != 2:
        print(
            f"[{cam_name}-Frame{folder_idx}] Warning: Invalid shape for img_points: {img_points.shape}. Expected Nx2. Returning empty."
        )
        return np.array([]), np.array([])
    if obj_points.ndim != 2 or obj_points.shape[1] != 3:
        print(
            f"[{cam_name}-Frame{folder_idx}] Warning: Invalid shape for obj_points: {obj_points.shape}. Expected Nx3. Returning empty."
        )
        return np.array([]), np.array([])

    initial_point_count = len(img_points)
    if initial_point_count == 0:
        return img_points, obj_points

    if img_points.shape[0] != obj_points.shape[0]:
        print(
            f"[{cam_name}-Frame{folder_idx}] Warning: Mismatched point counts before filtering: "
            f"{initial_point_count} vs {len(obj_points)}. Skipping filtering."
        )
        return img_points, obj_points

    output_dir = Path(f"pycalib/results/debugging/features/filtered_{cam_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Prepare for Visualization ---
    fig, axs = None, None
    if visualize_filtering:
        fig, axs = plt.subplots(1, 3, figsize=(21, 7), sharex=True, sharey=True)
        fig.suptitle(f"Feature Filtering Stages - Frame {folder_idx}", fontsize=16)

        if initial_point_count > 0:
            min_x, max_x = np.min(img_points[:, 0]), np.max(img_points[:, 0])
            min_y, max_y = np.min(img_points[:, 1]), np.max(img_points[:, 1])
            padding_x = (max_x - min_x) * 0.1 + 10
            padding_y = (max_y - min_y) * 0.1 + 10
            xlims = (min_x - padding_x, max_x + padding_x)
            ylims = (min_y - padding_y, max_y + padding_y)
        else:
            xlims = (0, 1280)
            ylims = (0, 720)  # Fallback

        axs[0].scatter(
            img_points[:, 0],
            img_points[:, 1],
            marker="o",
            s=50,
            edgecolors="gray",
            facecolors="none",
            label=f"Initial ({initial_point_count})",
        )
        axs[0].set_title("1. Initial Points")
        axs[0].set_ylabel("Y pixel")
        axs[0].legend(loc="upper right")
        axs[0].grid(True, linestyle="--", alpha=0.5)

    # --- Grid Consistency Filtering (RANSAC Homography) ---
    current_img_points = img_points.copy()
    current_obj_points = obj_points.copy()
    ransac_mask = None  # Mask relative to original img_points for visualization
    num_inliers, num_outliers = (
        initial_point_count,
        0,
    )  # Default for visualization if RANSAC fails/skipped

    if use_grid_filter and len(current_img_points) >= 4:
        obj_points_xy = current_obj_points[:, :2]
        try:
            H, mask = cv2.findHomography(
                obj_points_xy, current_img_points, cv2.RANSAC, ransac_thresh
            )
            if H is not None and mask is not None:
                ransac_mask = mask.ravel()
                current_inliers = np.sum(ransac_mask)
                if current_inliers > 0:
                    num_inliers = current_inliers
                    num_outliers = len(ransac_mask) - num_inliers
                    # Apply the mask to filter points for subsequent steps
                    current_img_points = current_img_points[ransac_mask == 1]
                    current_obj_points = current_obj_points[ransac_mask == 1]
                else:
                    # RANSAC found no inliers, treat all as inliers for plot, don't filter
                    ransac_mask = np.ones(initial_point_count, dtype=np.uint8)
                    num_inliers, num_outliers = initial_point_count, 0
            # else: findHomography failed, proceed without filtering / plotting outliers
        except cv2.error as e:
            # OpenCV error, proceed without filtering / plotting outliers
            print(
                f"[{cam_name}-Frame{folder_idx}] Warning: OpenCV error during findHomography: {e}. Skipping grid filtering."
            )
            ransac_mask = None  # Ensure mask is None so plot shows RANSAC skipped

    # Plot RANSAC Results (Subplot 2)
    if visualize_filtering:
        if use_grid_filter and ransac_mask is not None:
            inlier_points_initial = img_points[ransac_mask == 1]
            outlier_points_initial = img_points[ransac_mask == 0]

            if num_inliers > 0:
                axs[1].scatter(
                    inlier_points_initial[:, 0],
                    inlier_points_initial[:, 1],
                    marker="o",
                    s=50,
                    edgecolors="g",
                    facecolors="none",
                    linewidths=1.5,
                    label=f"Inliers ({num_inliers})",
                )
            if num_outliers > 0:
                axs[1].scatter(
                    outlier_points_initial[:, 0],
                    outlier_points_initial[:, 1],
                    marker="x",
                    s=40,
                    c="r",
                    linewidths=1.0,
                    label=f"Outliers ({num_outliers})",
                )
            axs[1].set_title(f"2. RANSAC Results (Thresh: {ransac_thresh}px)")
        else:
            axs[1].scatter(
                img_points[:, 0],
                img_points[:, 1],
                marker="o",
                s=50,
                edgecolors="gray",
                facecolors="none",
                label=f"Initial ({initial_point_count})",
            )
            axs[1].set_title("2. RANSAC Skipped / Failed")

        axs[1].legend(loc="upper right")
        axs[1].grid(True, linestyle="--", alpha=0.5)
        axs[1].set_xlabel("X pixel")

    # --- Radius-Based Filtering ---
    radius_filtered_points = current_img_points.copy()
    radius_filtered_obj_points = current_obj_points.copy()

    if radius_factor < 1.0 and len(radius_filtered_points) > 1:
        center = np.median(radius_filtered_points, axis=0)
        iteration = 0
        max_iterations = 5

        while iteration < max_iterations:
            iteration += 1
            num_points_before_iter = len(radius_filtered_points)

            if num_points_before_iter <= 1:
                break

            distances = np.sqrt(np.sum((radius_filtered_points - center) ** 2, axis=1))
            max_distance = np.max(distances)
            radius_threshold = max_distance * radius_factor

            mask = distances <= radius_threshold
            num_points_after_iter = np.sum(mask)

            if num_points_after_iter == 0:
                break  # Avoid removing all points
            if num_points_after_iter == num_points_before_iter:
                break  # Stable

            radius_filtered_points = radius_filtered_points[mask]
            radius_filtered_obj_points = radius_filtered_obj_points[mask]

            prev_center = center.copy()
            center = np.median(radius_filtered_points, axis=0)

            center_shift = np.sqrt(np.sum((center - prev_center) ** 2))
            if center_shift < 1.0:
                break  # Converged

        current_img_points = radius_filtered_points
        current_obj_points = radius_filtered_obj_points

    # --- Finalization and Plotting ---
    final_point_count = len(current_img_points)

    if visualize_filtering:
        if final_point_count > 0:
            axs[2].scatter(
                current_img_points[:, 0],
                current_img_points[:, 1],
                marker="o",
                s=60,
                edgecolors="b",
                linewidths=2.0,
                label=f"Final ({final_point_count})",
            )
        else:
            axs[2].text(
                0.5,
                0.5,
                "No points remaining",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axs[2].transAxes,
                fontsize=12,
                color="red",
            )

        axs[2].set_title(f"3. Final Points (Radius Factor: {radius_factor})")
        axs[2].legend(loc="upper right")
        axs[2].grid(True, linestyle="--", alpha=0.5)

        for ax in axs:
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            ax.invert_yaxis()
            ax.set_aspect("equal", adjustable="box")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_dir / f"frame_{folder_idx}.png")
        plt.close(fig)

    return current_img_points, current_obj_points


def save_cam_proj_feat_vis(cam_feature_data, proj_feature_data):
    """
    Visualize and save feature correspondences for camera and projector.
    """
    # Create output directories
    cam_viz_dir = Path("pycalib/results/debugging/features/2d-3d_camera")
    proj_viz_dir = Path("pycalib/results/debugging/features/2d-3d_projector")
    stereo_viz_dir = Path("pycalib/results/debugging/features/2d-2d_cam_proj")
    cam_viz_dir.mkdir(parents=True, exist_ok=True)
    proj_viz_dir.mkdir(parents=True, exist_ok=True)
    stereo_viz_dir.mkdir(parents=True, exist_ok=True)

    possible_common_frames = sorted(
        set(cam_feature_data.keys()) | set(proj_feature_data.keys())
    )

    proj_resolution = (768, 768)

    for frame_idx in possible_common_frames:
        cam_data = cam_feature_data.get(frame_idx, {})
        proj_data = proj_feature_data.get(frame_idx, {})

        # Get common data needed for visualizations
        cam_img = cam_data.get("ref_image")
        raw_cam_img_points = cam_data.get("imagePoints")
        raw_cam_obj_points = cam_data.get("objectPoints")
        raw_proj_img_points = proj_data.get("imagePoints")
        raw_proj_obj_points = proj_data.get("objectPoints")

        # --- Camera 2D-3D Visualization ---
        cam_correspondences = cam_data.get("2d_3d_corres")
        valid_cam_2d_3d = (
            cam_img is not None
            and isinstance(cam_correspondences, list)
            and len(cam_correspondences) > 0
        )

        ref_point_idx_2d3d = (
            -1
        )  # Index of reference point within the camera 2d-3d correspondence list
        ref_obj_point_tuple = None  # Reference object point as a tuple

        if valid_cam_2d_3d:
            # Extract points directly from camera correspondences list
            try:
                cam_img_points_from_corres = np.array(
                    [item["img_point"] for item in cam_correspondences],
                    dtype=np.float32,
                )
                cam_obj_points_from_corres = np.array(
                    [item["obj_point"] for item in cam_correspondences],
                    dtype=np.float32,
                )
                # Find the reference point index and its object point from camera data
                for i, item in enumerate(cam_correspondences):
                    if item.get("is_reference", False):
                        ref_point_idx_2d3d = i
                        ref_obj_point_tuple = tuple(
                            map(float, cam_obj_points_from_corres[i])
                        )
                        break

                # Check if extraction resulted in non-empty arrays of correct shape
                if not (
                    cam_img_points_from_corres.ndim == 2
                    and cam_img_points_from_corres.shape[1] == 2
                    and cam_obj_points_from_corres.ndim == 2
                    and cam_obj_points_from_corres.shape[1] == 3
                    and cam_img_points_from_corres.shape[0]
                    == cam_obj_points_from_corres.shape[0]
                    > 0
                ):
                    print(
                        f"Warning: Invalid data structure found in camera '2d_3d_corres' for frame {frame_idx}. Skipping Camera 2D-3D plot."
                    )
                    valid_cam_2d_3d = False
            except (KeyError, TypeError, ValueError) as e:
                print(
                    f"Warning: Error processing camera '2d_3d_corres' for frame {frame_idx}: {e}. Skipping Camera 2D-3D plot."
                )
                valid_cam_2d_3d = False

        if valid_cam_2d_3d:
            fig_cam, axs_cam = plt.subplots(1, 2, figsize=(18, 8))

            # Subplot 1: Camera Image with 2D Features
            if cam_img.ndim == 2:
                display_cam_img = cv2.cvtColor(cam_img, cv2.COLOR_GRAY2RGB)
            elif cam_img.ndim == 3 and cam_img.shape[2] == 1:
                display_cam_img = cv2.cvtColor(cam_img, cv2.COLOR_GRAY2RGB)
            elif cam_img.ndim == 3 and cam_img.shape[2] == 3:
                display_cam_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)
            else:
                display_cam_img = cam_img

            axs_cam[0].imshow(display_cam_img)
            x_coords_cam = cam_img_points_from_corres[:, 0]
            y_coords_cam = cam_img_points_from_corres[:, 1]

            # Plot all points first
            axs_cam[0].scatter(
                x_coords_cam,
                y_coords_cam,
                c="green",  # Changed color for better contrast with red ref point
                marker="o",
                s=30,
                alpha=0.7,
                label=f"Camera Features ({len(cam_img_points_from_corres)})",
            )
            # Highlight reference point if found
            if ref_point_idx_2d3d != -1:
                axs_cam[0].scatter(
                    x_coords_cam[ref_point_idx_2d3d],
                    y_coords_cam[ref_point_idx_2d3d],
                    c="red",
                    marker="o",
                    s=30,
                    alpha=0.9,
                    label="Reference Point",
                )

            # Add text labels (index only)
            for i, (x, y) in enumerate(zip(x_coords_cam, y_coords_cam)):
                point_color = "red" if i == ref_point_idx_2d3d else "white"
                bbox_color = "darkred" if i == ref_point_idx_2d3d else "black"
                axs_cam[0].text(
                    x + 5,
                    y + 5,
                    str(i),
                    fontsize=8,
                    color=point_color,
                    bbox=dict(facecolor=bbox_color, alpha=0.6, pad=0.1),
                )

            axs_cam[0].set_title(f"Frame {frame_idx}: Camera 2D Features")
            axs_cam[0].legend(loc="upper right")
            axs_cam[0].axis("on")

            # Subplot 2: 2D Object Points (X-Y Plane)
            obj_x = cam_obj_points_from_corres[:, 0]
            obj_y = cam_obj_points_from_corres[:, 1]
            # Assume Z is constant for the frame, get it from the first point
            z_coord = cam_obj_points_from_corres[0, 2]

            # Plot all object points
            axs_cam[1].scatter(
                obj_x,
                obj_y,
                c="green",  # Changed color
                marker="o",
                s=30,
                alpha=0.7,
                label=f"Object Points ({len(cam_obj_points_from_corres)})",
            )
            # Highlight corresponding reference object point if found
            if ref_point_idx_2d3d != -1:
                axs_cam[1].scatter(
                    obj_x[ref_point_idx_2d3d],
                    obj_y[ref_point_idx_2d3d],
                    c="red",
                    marker="o",
                    s=30,
                    alpha=0.9,
                    label="Reference Point",
                )

            # Add text labels: index above, coordinates below
            y_offset = 0.15  # Adjust this offset based on your coordinate scale
            for i, (x, y) in enumerate(zip(obj_x, obj_y)):
                text_color = "red" if i == ref_point_idx_2d3d else "black"
                # Index above the point
                axs_cam[1].text(
                    x,
                    y - y_offset,  # Position above
                    str(i),
                    fontsize=8,
                    color=text_color,
                    ha="center",  # Horizontal alignment centered
                    va="bottom",  # Vertical alignment bottom (anchor text above point)
                )
                # Coordinates below the point
                coord_text = f"({x:.1f}, {y:.1f})"
                axs_cam[1].text(
                    x,
                    y + y_offset,  # Position below
                    coord_text,
                    fontsize=6,  # Smaller font for coordinates
                    color=text_color,
                    ha="center",  # Horizontal alignment centered
                    va="top",  # Vertical alignment top (anchor text below point)
                )

            axs_cam[1].set_xlabel("X (mm)")
            axs_cam[1].set_ylabel("Y (mm)")
            # Add Z-coordinate to the title
            axs_cam[1].set_title(
                f"Frame {frame_idx}: Object Points (Z={z_coord:.2f}mm)"
            )
            axs_cam[1].legend(loc="upper right")
            axs_cam[1].invert_yaxis()  # Invert Y axis for object points plot
            axs_cam[1].set_aspect(
                "equal", adjustable="box"
            )  # Ensure correct aspect ratio for grid
            axs_cam[1].grid(True)

            plt.suptitle(
                f"Camera 2D Features vs Object XY Plane - Frame {frame_idx}",
                fontsize=16,
            )
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(cam_viz_dir / f"frame_{frame_idx}_cam_2d_3d.png")
            plt.close(fig_cam)

        # --- Projector 2D-3D Visualization ---
        valid_proj_2d_3d = (
            isinstance(raw_proj_img_points, np.ndarray)
            and isinstance(raw_proj_obj_points, np.ndarray)
            and raw_proj_img_points.size > 0
            and raw_proj_obj_points.size > 0
            and raw_proj_img_points.ndim == 2
            and raw_proj_img_points.shape[1] == 2
            and raw_proj_obj_points.ndim == 2
            and raw_proj_obj_points.shape[1] == 3
            and raw_proj_img_points.shape[0] == raw_proj_obj_points.shape[0]
        )

        ref_point_idx_proj = (
            -1
        )  # Index of reference point within the projector data arrays

        if valid_proj_2d_3d and ref_obj_point_tuple:
            try:
                # Find the index of the reference object point in the raw projector object points
                proj_obj_tuples_raw = [
                    tuple(map(float, point)) for point in raw_proj_obj_points
                ]
                if ref_obj_point_tuple in proj_obj_tuples_raw:
                    ref_point_idx_proj = proj_obj_tuples_raw.index(ref_obj_point_tuple)
            except ValueError:
                ref_point_idx_proj = -1  # Reference point not found in projector data

        if valid_proj_2d_3d:
            fig_proj, axs_proj = plt.subplots(1, 2, figsize=(18, 8))

            # Subplot 1: Projector 2D Features
            axs_proj[0].set_facecolor("black")
            axs_proj[0].set_xlim(0, proj_resolution[0])
            axs_proj[0].set_ylim(proj_resolution[1], 0)  # Invert Y for image coords
            x_coords_proj = raw_proj_img_points[:, 0]
            y_coords_proj = raw_proj_img_points[:, 1]

            # Plot all projector points
            axs_proj[0].scatter(
                x_coords_proj,
                y_coords_proj,
                c="magenta",
                marker="o",
                s=30,
                alpha=0.7,
                label=f"Projector Features ({len(raw_proj_img_points)})",
            )
            # Highlight reference point if found
            if ref_point_idx_proj != -1:
                axs_proj[0].scatter(
                    x_coords_proj[ref_point_idx_proj],
                    y_coords_proj[ref_point_idx_proj],
                    c="red",
                    marker="o",
                    s=30,
                    alpha=0.9,
                    label="Reference Point",
                )

            # Add text labels (index only)
            for i, (x, y) in enumerate(zip(x_coords_proj, y_coords_proj)):
                point_color = "red" if i == ref_point_idx_proj else "white"
                bbox_color = "darkred" if i == ref_point_idx_proj else "gray"
                axs_proj[0].text(
                    x + 5,
                    y + 5,
                    str(i),
                    fontsize=8,
                    color=point_color,
                    bbox=dict(facecolor=bbox_color, alpha=0.6, pad=0.1),
                )

            axs_proj[0].set_title(f"Frame {frame_idx}: Projector 2D Features")
            axs_proj[0].legend(loc="upper right")
            axs_proj[0].set_xlabel("X (pixels)")
            axs_proj[0].set_ylabel("Y (pixels)")
            axs_proj[0].set_aspect("equal", adjustable="box")
            axs_proj[0].axis("on")  # Keep axis lines

            # Subplot 2: Corresponding Object Points (X-Y Plane)
            obj_x_proj = raw_proj_obj_points[:, 0]
            obj_y_proj = raw_proj_obj_points[:, 1]
            z_coord_proj = raw_proj_obj_points[
                0, 2
            ]  # Assume Z is constant for the frame

            # Plot all object points corresponding to projector points
            axs_proj[1].scatter(
                obj_x_proj,
                obj_y_proj,
                c="magenta",  # Match projector color
                marker="o",
                s=30,
                alpha=0.7,
                label=f"Object Points ({len(raw_proj_obj_points)})",
            )
            # Highlight corresponding reference object point if found
            if ref_point_idx_proj != -1:
                axs_proj[1].scatter(
                    obj_x_proj[ref_point_idx_proj],
                    obj_y_proj[ref_point_idx_proj],
                    c="red",
                    marker="o",
                    s=30,
                    alpha=0.9,
                    label="Reference Point",
                )

            # Add text labels: index above, coordinates below
            y_offset = 0.15  # Adjust this offset based on your coordinate scale
            for i, (x, y) in enumerate(zip(obj_x_proj, obj_y_proj)):
                text_color = "red" if i == ref_point_idx_proj else "black"
                # Index above the point
                axs_proj[1].text(
                    x,
                    y - y_offset,  # Position above
                    str(i),
                    fontsize=8,
                    color=text_color,
                    ha="center",  # Horizontal alignment centered
                    va="bottom",  # Vertical alignment bottom (anchor text above point)
                )
                # Coordinates below the point
                coord_text = f"({x:.1f}, {y:.1f})"
                axs_proj[1].text(
                    x,
                    y + y_offset,  # Position below
                    coord_text,
                    fontsize=6,  # Smaller font for coordinates
                    color=text_color,
                    ha="center",  # Horizontal alignment centered
                    va="top",  # Vertical alignment top (anchor text below point)
                )

            axs_proj[1].set_xlabel("X (mm)")
            axs_proj[1].set_ylabel("Y (mm)")
            axs_proj[1].set_title(
                f"Frame {frame_idx}: Corresponding Object Points (Z={z_coord_proj:.2f}mm)"
            )
            axs_proj[1].legend(loc="upper right")
            axs_proj[1].invert_yaxis()
            axs_proj[1].set_aspect("equal", adjustable="box")
            axs_proj[1].grid(True)

            plt.suptitle(
                f"Projector 2D Features vs Object XY Plane - Frame {frame_idx}",
                fontsize=16,
            )
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(
                proj_viz_dir / f"frame_{frame_idx}_proj_2d_3d.png"
            )  # Save to projector dir
            plt.close(fig_proj)
        elif not valid_proj_2d_3d:
            pass  # Optionally print a warning if needed
        # print(f"Skipping projector 2D-3D plot for frame {frame_idx}: Incomplete or invalid projector data.")

        # --- Stereo 2D-2D Visualization ---
        # Check if basic camera data exists before attempting alignment
        if (
            cam_img is None
            or not isinstance(raw_cam_img_points, np.ndarray)
            or not isinstance(raw_cam_obj_points, np.ndarray)
            or not valid_proj_2d_3d  # Reuse projector data check
            or raw_cam_img_points.size == 0
            or raw_cam_obj_points.size == 0
            # Projector points already checked in valid_proj_2d_3d
            or raw_cam_img_points.shape[0] != raw_cam_obj_points.shape[0]
            # Projector shapes already checked in valid_proj_2d_3d
        ):
            # print(f"Skipping stereo plot for frame {frame_idx}: Incomplete or mismatched raw data.")
            continue  # Skip to next frame if essential raw data is missing

        # Align based on common object points
        final_cam_img_points = np.array([])
        final_proj_img_points = np.array([])
        valid_stereo_2d_2d = False
        ref_point_idx_stereo = (
            -1
        )  # Index of ref point within the *aligned* stereo lists

        try:
            cam_obj_tuples = [tuple(map(float, point)) for point in raw_cam_obj_points]
            proj_obj_tuples = [
                tuple(map(float, point)) for point in raw_proj_obj_points
            ]

            common_obj_tuples_set = set(cam_obj_tuples) & set(proj_obj_tuples)

            if not common_obj_tuples_set:
                # print(f"Skipping stereo plot for frame {frame_idx}: No common object points found.")
                continue  # Skip to next frame

            # Maintain order for consistent indexing
            common_obj_tuples_list = sorted(list(common_obj_tuples_set))

            # Create masks for filtering image points based on common object points
            # Need to map original indices to know which raw points correspond
            cam_orig_indices_map = {
                obj_tuple: i for i, obj_tuple in enumerate(cam_obj_tuples)
            }
            proj_orig_indices_map = {
                obj_tuple: i for i, obj_tuple in enumerate(proj_obj_tuples)
            }

            cam_indices_to_keep = [
                cam_orig_indices_map[obj_tuple]
                for obj_tuple in common_obj_tuples_list
                if obj_tuple in cam_orig_indices_map
            ]
            proj_indices_to_keep = [
                proj_orig_indices_map[obj_tuple]
                for obj_tuple in common_obj_tuples_list
                if obj_tuple in proj_orig_indices_map
            ]

            # Check if the number of points matches after finding indices (should match common_obj_tuples_list size)
            if len(cam_indices_to_keep) != len(common_obj_tuples_list) or len(
                proj_indices_to_keep
            ) != len(common_obj_tuples_list):
                print(
                    f"Warning: Mismatch in indices during stereo alignment for frame {frame_idx}. Skipping."
                )
                continue

            # Build final aligned arrays using the identified indices
            final_cam_img_points = raw_cam_img_points[cam_indices_to_keep].astype(
                np.float32
            )
            final_proj_img_points = raw_proj_img_points[proj_indices_to_keep].astype(
                np.float32
            )

            # Find the index of the original reference point within the common/aligned list
            if ref_obj_point_tuple and ref_obj_point_tuple in common_obj_tuples_list:
                try:
                    ref_point_idx_stereo = common_obj_tuples_list.index(
                        ref_obj_point_tuple
                    )
                except ValueError:
                    ref_point_idx_stereo = (
                        -1
                    )  # Should not happen if check above passed, but safety first

            # Validate the final aligned data
            valid_stereo_2d_2d = (
                final_cam_img_points.ndim == 2
                and final_cam_img_points.shape[1] == 2
                and final_proj_img_points.ndim == 2
                and final_proj_img_points.shape[1] == 2
                and final_cam_img_points.shape[0] == final_proj_img_points.shape[0]
                and final_cam_img_points.shape[0]
                > 0  # Ensure we have at least one point
            )

        except Exception as e:
            print(
                f"Error during stereo alignment for frame {frame_idx}: {e}. Skipping plot."
            )
            valid_stereo_2d_2d = False

        if valid_stereo_2d_2d:
            proj_resolution = (768, 768)
            fig_stereo, axs_stereo = plt.subplots(1, 2, figsize=(18, 9))

            # Subplot 1: Camera Image with ALIGNED 2D Features
            if cam_img.ndim == 2:
                display_cam_img = cv2.cvtColor(cam_img, cv2.COLOR_GRAY2RGB)
            elif cam_img.ndim == 3 and cam_img.shape[2] == 1:
                display_cam_img = cv2.cvtColor(cam_img, cv2.COLOR_GRAY2RGB)
            elif cam_img.ndim == 3 and cam_img.shape[2] == 3:
                display_cam_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)
            else:
                display_cam_img = cam_img  # Fallback

            axs_stereo[0].imshow(display_cam_img)
            # Use the FINAL aligned points for plotting
            x_coords_cam_stereo = final_cam_img_points[:, 0]
            y_coords_cam_stereo = final_cam_img_points[:, 1]

            # Plot all points
            axs_stereo[0].scatter(
                x_coords_cam_stereo,
                y_coords_cam_stereo,
                c="green",
                marker="o",
                s=30,
                alpha=0.7,
                label=f"Camera Features ({len(final_cam_img_points)})",
            )
            # Highlight reference point if found in aligned list
            if ref_point_idx_stereo != -1:
                axs_stereo[0].scatter(
                    x_coords_cam_stereo[ref_point_idx_stereo],
                    y_coords_cam_stereo[ref_point_idx_stereo],
                    c="red",
                    marker="o",
                    s=30,
                    alpha=0.9,
                    label="Reference Point",
                )

            # Add text labels (index only)
            for i, (x, y) in enumerate(zip(x_coords_cam_stereo, y_coords_cam_stereo)):
                point_color = "red" if i == ref_point_idx_stereo else "white"
                bbox_color = "darkred" if i == ref_point_idx_stereo else "gray"
                axs_stereo[0].text(
                    x + 5,
                    y + 5,
                    str(i),
                    fontsize=8,
                    color=point_color,
                    bbox=dict(facecolor=bbox_color, alpha=0.6, pad=0.1),
                )
            axs_stereo[0].set_title(
                f"Frame {frame_idx}: Camera 2D Features (Stereo View)"
            )
            axs_stereo[0].legend(loc="upper right")
            axs_stereo[0].axis("on")

            # Subplot 2: Projector 2D Features (ALIGNED)
            axs_stereo[1].set_facecolor("black")
            axs_stereo[1].set_xlim(0, proj_resolution[0])
            axs_stereo[1].set_ylim(proj_resolution[1], 0)  # Invert Y for image coords
            # Use the FINAL aligned points for plotting
            x_coords_proj_stereo = final_proj_img_points[:, 0]
            y_coords_proj_stereo = final_proj_img_points[:, 1]

            # Plot all points
            axs_stereo[1].scatter(
                x_coords_proj_stereo,
                y_coords_proj_stereo,
                c="magenta",
                marker="o",
                s=30,
                alpha=0.7,
                label=f"Projector Features ({len(final_proj_img_points)})",
            )
            # Highlight reference point if found in aligned list
            if ref_point_idx_stereo != -1:
                axs_stereo[1].scatter(
                    x_coords_proj_stereo[ref_point_idx_stereo],
                    y_coords_proj_stereo[ref_point_idx_stereo],
                    c="red",
                    marker="o",
                    s=30,
                    alpha=0.9,
                    label="Reference Point",
                )

            # Add text labels (index only)
            # Use the same index 'i' to show correspondence
            for i, (x, y) in enumerate(zip(x_coords_proj_stereo, y_coords_proj_stereo)):
                point_color = "red" if i == ref_point_idx_stereo else "white"
                bbox_color = "darkred" if i == ref_point_idx_stereo else "gray"
                axs_stereo[1].text(
                    x + 5,
                    y + 5,
                    str(i),
                    fontsize=8,
                    color=point_color,
                    bbox=dict(facecolor=bbox_color, alpha=0.6, pad=0.1),
                )
            axs_stereo[1].set_title(
                f"Frame {frame_idx}: Corresponding Projector 2D Features"
            )
            axs_stereo[1].legend(loc="upper right")
            axs_stereo[1].set_aspect("equal", adjustable="box")
            axs_stereo[1].set_xlabel("X (pixels)")
            axs_stereo[1].set_ylabel("Y (pixels)")

            plt.suptitle(
                f"Stereo Camera-Projector 2D Correspondence - Frame {frame_idx}",
                fontsize=16,
            )
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(stereo_viz_dir / f"frame_{frame_idx}_stereo_2d_2d.png")
            plt.close(fig_stereo)
        # If valid_stereo_2d_2d is false after alignment attempt, the plot is skipped.


def rvec_to_euler(rvec):
    """
    Convert rotation vector to Euler angles (roll, pitch, yaw) in degrees.
    Uses ZYX convention (yaw -> pitch -> roll).
    """
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Extract angles - using math to avoid potential gimbal lock issues
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))

    if np.abs(pitch) > np.pi / 2 - 1e-8:  # Close to ±90°
        # At the poles, gimbal lock occurs, so we need to handle differently
        # Choose a convention: set yaw = 0, and compute roll
        yaw = 0
        roll = np.arctan2(R[0, 1], R[1, 1])
    else:
        # Normal case
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])

    # Convert to degrees
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)

    return [roll_deg, pitch_deg, yaw_deg]
