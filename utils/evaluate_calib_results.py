import pickle
from pathlib import Path

import cv2
from matplotlib import pyplot as plt
import numpy as np

from pycalib.feature_processing.extract_features import ImageProcessor
from pycalib.feature_processing.feature_extractor_cv import FeatureExtractorCV
from pycalib.optimization.optimizer_configs import Extrinsics, Intrinsics


def load_images(folder_path: str):
    ref_image = None
    folder_path = Path(folder_path)
    # img_files = sorted(folder_path.iterdir(), key=lambda x: int(x.stem))
    img_list = []
    for img in folder_path.iterdir():
        if img.name != "img0.npy" and img.suffix == "img.npy":
            img = ImageProcessor.load_image(img)
            # plt.imshow(img)
            # plt.show()
            img_list.append(img)
        elif img.name == "img0.npy":
            ref_image = ImageProcessor.load_image(img)

    return ref_image, img_list


def extract_device_params(calib_result: dict) -> dict:
    resolution = calib_result["image_size"]
    intrinsics: Intrinsics = calib_result["intrinsics"]
    extrinsics: Extrinsics = calib_result["extrinsics"]

    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.cx
    cy = intrinsics.cy
    dist_coeffs = intrinsics.dist_coeffs.flatten().tolist()

    rvec = extrinsics.rvec.flatten()
    tvec = extrinsics.tvec.flatten().reshape(3, 1)

    K = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    R, _ = cv2.Rodrigues(rvec)
    Rt = np.hstack((R, tvec)).tolist()

    undist_coeffs = [-d for d in dist_coeffs]

    return {
        "K": K,
        "Rt": Rt,
        "distortion": dist_coeffs,
        "resolution": resolution,
        "undistortion": undist_coeffs,
    }


def eval_undistortion(
    ori_image: np.ndarray,
    cam_params: dict,
):
    K = np.array(cam_params["K"])
    dist_coeffs = np.array(cam_params["distortion"])
    undistorted_image = cv2.undistort(ori_image, K, dist_coeffs)

    # --- Visualization ---
    h, w = ori_image.shape[:2]
    num_lines = 10  # Number of grid lines

    # Create copies to draw on
    ori_image_with_lines = ori_image.copy()
    undistorted_image_with_lines = undistorted_image.copy()

    # Draw vertical lines
    for i in range(1, num_lines):
        x = int(w * i / num_lines)
        cv2.line(
            ori_image_with_lines, (x, 0), (x, h - 1), (0, 255, 0), 1
        )  # Green lines
        cv2.line(undistorted_image_with_lines, (x, 0), (x, h - 1), (0, 255, 0), 1)

    # Draw horizontal lines
    for i in range(1, num_lines):
        y = int(h * i / num_lines)
        cv2.line(
            ori_image_with_lines, (0, y), (w - 1, y), (0, 255, 0), 1
        )  # Green lines
        cv2.line(undistorted_image_with_lines, (0, y), (w - 1, y), (0, 255, 0), 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(ori_image_with_lines)
    plt.title("Original Image with Grid")
    plt.subplot(1, 2, 2)
    plt.imshow(undistorted_image_with_lines)
    plt.title("Undistorted Image with Grid")
    plt.show()
    # --- End Visualization ---


def eval_3d_reconstruction(
    ori_image: np.ndarray,
    cam_params: dict,
    image_points: np.ndarray,
    object_points: np.ndarray,
    image_idx: int,
    target_spacing_mm: float,
):
    # 1. Extract parameters
    K = np.array(cam_params["K"])
    dist_coeffs = np.array(cam_params["distortion"])
    Rt = np.array(cam_params["Rt"])
    R = Rt[:, :3]
    t = Rt[:, 3]
    # World Z coordinate of the target plane for this image
    target_z_world = 0.25 * image_idx

    # Ensure image_points has the shape (N, 1, 2) for OpenCV functions
    if image_points.ndim == 2:
        # Handles shape (N, 2)
        image_points = image_points.reshape(-1, 1, 2).astype(np.float32)
    elif image_points.shape[1:] != (1, 2):
        # Handles other potential shapes like (1, N, 2) if necessary
        image_points = image_points.reshape(-1, 1, 2).astype(np.float32)
    else:
        image_points = image_points.astype(np.float32)  # Ensure correct dtype

    # Ensure object_points has shape (N, 3)
    if object_points.ndim == 2 and object_points.shape[1] == 3:
        object_points = object_points.astype(np.float32)
    else:
        # Handle potential reshaping if needed, although (N,3) is expected
        # Example: Reshape from (N, 1, 3)
        object_points = object_points.reshape(-1, 3).astype(np.float32)

    # 2. Undistort 2D image points to normalized image coordinates (x', y')
    # Using P=identity returns normalized coordinates directly
    normalized_coords = cv2.undistortPoints(image_points, K, dist_coeffs, P=np.eye(3))
    normalized_coords = normalized_coords.reshape(-1, 2)  # Shape (N, 2)

    # 3. Reconstruct 3D points in Camera Coordinates using Plane Constraint
    # Plane equation in world: Z_w = target_z_world
    # Normal vector of the plane in world coords: n_w = [0, 0, 1]^T
    # A point on the plane in world coords: P0_w = [0, 0, target_z_world]^T

    # Transform plane normal and point to camera coordinates:
    # P_c = R * P_w + t
    # Normal vector in camera coords (direction of World Z-axis in Camera frame)
    n_c = R[:, 2]
    # A point on the plane in camera coords
    P0_c = R @ np.array([0, 0, target_z_world], dtype=np.float32) + t

    # Calculate the scalar product n_c . P0_c (part of the plane equation constant)
    nc_dot_P0c = np.dot(n_c, P0_c)

    # Homogeneous normalized coordinates [x', y', 1] for all points
    homogeneous_norm_coords = np.hstack(
        (normalized_coords, np.ones((normalized_coords.shape[0], 1)))
    )  # Shape (N, 3)

    # Calculate the denominator term n_c . [x', y', 1]^T for each point
    # This comes from substituting P_c = Z_c * [x', y', 1]^T into the plane equation
    # n_c . (P_c - P0_c) = 0 => n_c . (Z_c * [x', y', 1]^T) - n_c . P0_c = 0
    # => Z_c * (n_c . [x', y', 1]^T) = n_c . P0_c
    denominator = homogeneous_norm_coords @ n_c  # Shape (N,)

    # Calculate Z_c (depth) for each point
    # Avoid division by zero or very small numbers
    valid_indices = np.abs(denominator) > 1e-6
    Z_c = np.full(denominator.shape[0], np.nan, dtype=np.float32)
    Z_c[valid_indices] = nc_dot_P0c / denominator[valid_indices]

    # Reconstruct 3D points in camera coordinates: P_c = Z_c * [x', y', 1]^T
    reconstructed_points_cam = Z_c[:, np.newaxis] * homogeneous_norm_coords

    # Filter out points where Z_c could not be reliably calculated
    reconstructed_points_cam = reconstructed_points_cam[valid_indices]
    # Keep corresponding ground truth points consistent
    valid_object_points = object_points[valid_indices]

    # 4. Transform Ground Truth object points (World Coords) to Camera Coordinates
    # object_points shape is (N, 3), R is (3, 3), t is (3,)
    # P_c = R * P_w + t => P_c^T = P_w^T * R^T + t^T
    # Using (N, 3) @ (3, 3) + (1, 3) broadcasting
    object_points_cam = (R @ valid_object_points.T).T + t.reshape(1, 3)

    # --- Calculate Reconstruction Error ---
    if reconstructed_points_cam.shape[0] > 0:
        # Calculate Euclidean distance for each point pair
        distances = np.linalg.norm(reconstructed_points_cam - object_points_cam, axis=1)
        # Calculate the mean distance (error)
        mean_error_mm = np.mean(distances)
        print(f"Mean Reconstruction Error (Camera Frame): {mean_error_mm:.4f} mm")
    else:
        print("Warning: No valid points found for error calculation.")
    # --- End Error Calculation ---

    # --- Start 2D Plane Visualization ---
    if (
        reconstructed_points_cam.shape[0] >= 3
    ):  # Need at least 3 points for PCA/plane fitting
        # 1. Fit Plane using PCA on reconstructed camera points
        centroid = np.mean(reconstructed_points_cam, axis=0)
        centered_points = reconstructed_points_cam - centroid
        # Use SVD for potentially better numerical stability than covariance matrix
        try:
            _, _, vh = np.linalg.svd(centered_points)
            # vh contains principal components as rows, sorted from largest variance
            plane_normal = vh[
                2, :
            ]  # Last row corresponds to smallest variance -> normal
            basis_v1 = vh[0, :]  # First row -> first basis vector in plane
            basis_v2 = vh[1, :]  # Second row -> second basis vector in plane
        except np.linalg.LinAlgError:
            print("Warning: SVD did not converge for plane fitting. Skipping 2D plot.")
            reconstructed_points_2d = None  # Flag to skip plotting

        if "basis_v1" in locals():  # Check if SVD was successful
            # 2. Project 3D points onto the 2D plane basis
            reconstructed_points_2d = np.zeros((reconstructed_points_cam.shape[0], 2))
            for i, p in enumerate(centered_points):
                reconstructed_points_2d[i, 0] = np.dot(p, basis_v1)  # u coordinate
                reconstructed_points_2d[i, 1] = np.dot(p, basis_v2)  # v coordinate

            # 3. Identify adjacent points using valid_object_points and target_spacing_mm
            adjacent_pairs = []  # List of tuples: (index1, index2)
            num_valid_points = valid_object_points.shape[0]
            # Increase tolerance slightly for floating point comparisons
            tolerance = target_spacing_mm * 0.15

            # Check distances between all pairs in the original world grid points
            for i in range(num_valid_points):
                for j in range(i + 1, num_valid_points):
                    # Use valid_object_points which correspond to the reconstructed points
                    dist_3d_world = np.linalg.norm(
                        valid_object_points[i] - valid_object_points[j]
                    )
                    # Check if distance is close to the target spacing
                    if abs(dist_3d_world - target_spacing_mm) < tolerance:
                        adjacent_pairs.append((i, j))

            # 4. Calculate distances between adjacent points in the 2D projection
            adjacent_distances = []  # List of distances corresponding to adjacent_pairs
            for idx1, idx2 in adjacent_pairs:
                dist_2d = np.linalg.norm(
                    reconstructed_points_2d[idx1] - reconstructed_points_2d[idx2]
                )
                adjacent_distances.append(dist_2d)

            # 5. Create 2D Plot
            fig_2d, ax_2d = plt.subplots(figsize=(9, 9))
            ax_2d.scatter(
                reconstructed_points_2d[:, 0],
                reconstructed_points_2d[:, 1],
                marker="o",
                label="Reconstructed Points (2D Projection)",
                s=25,  # Slightly larger points
                edgecolors="k",  # Add edge color for visibility
                linewidths=0.5,
            )

            # Store label positions and bounding boxes to check for overlap
            label_bboxes = []

            # Plot lines and labels for adjacent points
            for k, (idx1, idx2) in enumerate(adjacent_pairs):
                p1 = reconstructed_points_2d[idx1]
                p2 = reconstructed_points_2d[idx2]
                dist = adjacent_distances[k]

                # Draw line
                ax_2d.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    color="gray",
                    linestyle="-",
                    linewidth=0.8,
                    alpha=0.6,
                )

                # Calculate label position (midpoint) and text
                mid_point = (p1 + p2) / 2.0
                label_text = f"{dist:.2f}"

                # Estimate text size (this is approximate)
                text_obj = ax_2d.text(
                    mid_point[0],
                    mid_point[1],
                    label_text,
                    fontsize=7,
                    color="blue",
                    ha="center",
                    va="center",
                    alpha=0,
                )  # Render initially invisible to get size
                bbox = text_obj.get_window_extent(fig_2d.canvas.get_renderer())
                # Transform bbox from display coords to data coords
                bbox_data = bbox.transformed(ax_2d.transData.inverted())

                # --- Basic Overlap Avoidance ---
                # Check overlap with previous labels' bounding boxes
                is_overlapping = False
                for existing_bbox in label_bboxes:
                    if bbox_data.overlaps(existing_bbox):
                        is_overlapping = True
                        break

                # If not overlapping, place the label and store its bbox
                if not is_overlapping:
                    text_obj.set_alpha(1.0)  # Make visible
                    label_bboxes.append(bbox_data)
                else:
                    text_obj.remove()  # Remove the overlapping text object
                    # Optional: Could try alternative positions here
                    # print(f"Skipping label for pair ({idx1}, {idx2}) due to overlap.")

            ax_2d.set_xlabel("Plane Basis Vector 1 (u) [mm]")
            ax_2d.set_ylabel("Plane Basis Vector 2 (v) [mm]")
            ax_2d.set_title(
                f"2D Projection onto Best-Fit Plane (Image {image_idx})\nAdjacent Distances (Target ≈ {target_spacing_mm:.2f} mm)"
            )
            ax_2d.set_aspect("equal", adjustable="box")
            ax_2d.grid(True, linestyle="--", alpha=0.5)
            # ax_2d.legend() # Legend might be redundant

    else:
        print(
            "Warning: Cannot perform 2D plane visualization with fewer than 3 points."
        )
    # --- End 2D Plane Visualization ---

    # 5. Visualize in 3D (Camera Coordinate System)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot the reconstructed points
    ax.scatter(
        reconstructed_points_cam[:, 0],
        reconstructed_points_cam[:, 1],
        reconstructed_points_cam[:, 2],
        marker="o",
        label=f"Reconstructed (N={reconstructed_points_cam.shape[0]})",
        s=20,  # Point size
        alpha=0.7,
    )

    # Scatter plot the ground truth points transformed to camera coordinates
    ax.scatter(
        object_points_cam[:, 0],
        object_points_cam[:, 1],
        object_points_cam[:, 2],
        marker="x",
        label=f"Ground Truth (N={object_points_cam.shape[0]})",
        s=20,  # Point size
        alpha=0.7,
    )

    # Optional: Draw lines connecting corresponding points for error visualization
    # for i in range(reconstructed_points_cam.shape[0]):
    #     ax.plot(
    #         [reconstructed_points_cam[i, 0], object_points_cam[i, 0]],
    #         [reconstructed_points_cam[i, 1], object_points_cam[i, 1]],
    #         [reconstructed_points_cam[i, 2], object_points_cam[i, 2]],
    #         color='gray', linestyle='--', linewidth=0.5, alpha=0.5
    #     )

    # Set labels and title
    ax.set_xlabel("X (Camera Frame)")
    ax.set_ylabel("Y (Camera Frame)")
    ax.set_zlabel("Z (Camera Frame)")
    ax.set_title(
        f"3D Points in Camera Frame (Image {image_idx}, World Z={target_z_world:.2f})"
    )

    # Calculate bounds for equal aspect ratio and axis drawing
    all_points = np.vstack((reconstructed_points_cam, object_points_cam))
    # Include origin for bounds calculation if it's far from points
    all_points_with_origin = np.vstack((all_points, np.array([[0, 0, 0]])))
    max_range = (
        np.array(
            [
                all_points_with_origin[:, 0].max() - all_points_with_origin[:, 0].min(),
                all_points_with_origin[:, 1].max() - all_points_with_origin[:, 1].min(),
                all_points_with_origin[:, 2].max() - all_points_with_origin[:, 2].min(),
            ]
        ).max()
        / 2.0
    )
    # Handle cases where max_range might be zero or very small
    if max_range < 1e-6:
        max_range = 1.0  # Default range if points are coincident

    mid_x = (
        all_points_with_origin[:, 0].max() + all_points_with_origin[:, 0].min()
    ) * 0.5
    mid_y = (
        all_points_with_origin[:, 1].max() + all_points_with_origin[:, 1].min()
    ) * 0.5
    mid_z = (
        all_points_with_origin[:, 2].max() + all_points_with_origin[:, 2].min()
    ) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Draw Camera Coordinate System Axes
    axis_length = max_range * 0.3  # Adjust length as needed
    origin = [0, 0, 0]
    # X-axis (Red)
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        axis_length,
        0,
        0,
        color="red",
        arrow_length_ratio=0.1,
    )
    ax.text(origin[0] + axis_length * 1.1, origin[1], origin[2], "X", color="red")
    # Y-axis (Green)
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        0,
        axis_length,
        0,
        color="green",
        arrow_length_ratio=0.1,
    )
    ax.text(origin[0], origin[1] + axis_length * 1.1, origin[2], "Y", color="green")
    # Z-axis (Blue)
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        0,
        0,
        axis_length,
        color="blue",
        arrow_length_ratio=0.1,
    )
    ax.text(origin[0], origin[1], origin[2] + axis_length * 1.1, "Z", color="blue")

    ax.legend()  # Call legend after adding all plot elements

    # Alternatively, use matplotlib's built-in aspect setting if preferred
    # ax.set_aspect('equal', adjustable='box')

    plt.show()

def eval_reprojection():
    image_path = Path("pycalib/data/image_sources/image_data_cv")

    extractor = FeatureExtractorCV()

    extractor.load_images(image_path)

    if not extractor.ref_images_dict:
        print("No reference images loaded. Exiting.")
        return
    for folder_idx, ref_image in extractor.ref_images_dict.items():
        img = ref_image

        features = extractor.detect_grid_features(img)

        
        
if __name__ == "__main__":
    cam_params_pkl = "pycalib/data/cache/calib_result_Camera.pkl"
    cam_params_path = Path(cam_params_pkl)

    with open(cam_params_path, "rb") as f:
        cam_calib_result = pickle.load(f)

    cam_params = extract_device_params(cam_calib_result)

    image_idx = 21
    # Load only the specific image needed (using its index in the filename pattern)
    # Assuming file names like 'img21.npy' exist in the folder
    img_folder = Path(f"pycalib/data/image_sources/image_data_freq25/{image_idx}")
    img_path = img_folder / f"img{image_idx}.npy"
    if img_path.exists():
        ref_image = ImageProcessor.load_image(img_path)
    else:
        # Fallback or error handling if the specific image isn't found
        # For now, trying to load img0.npy as a fallback if needed
        ref_image, _ = load_images(
            f"pycalib/data/image_sources/image_data_freq25/{image_idx}"
        )
        print(f"Warning: Could not find {img_path}. Loaded img0.npy instead.")

    # Evaluate Undistortion (Optional)
    # eval_undistortion(ref_image, cam_params)

    # Evaluate 3D Reconstruction
    with open("pycalib/data/cache/filtered_cam_feature_data.pkl", "rb") as f:
        camera_feat_data = pickle.load(f)

    # Ensure data for the selected image_idx exists
    if image_idx in camera_feat_data:
        image_points = camera_feat_data[image_idx]["imagePoints"]
        object_points = camera_feat_data[image_idx]["objectPoints"]
        # Define the expected spacing (assuming it's consistent)
        TARGET_SPACING = 0.5  # mm

        # Corrected function call (positional args first)
        eval_3d_reconstruction(
            ref_image,
            cam_params,
            image_points,
            object_points,
            image_idx,
            target_spacing_mm=TARGET_SPACING,  # Pass spacing
        )
    else:
        print(f"Error: Feature data for image_idx {image_idx} not found in cache file.")
