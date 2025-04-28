from pathlib import Path
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any, Set
from scipy.spatial import KDTree

from pycalib.feature_processing.feature_extractor import ImageProcessor


# --- Configuration Dataclasses ---
@dataclass
class PreprocessingParams:
    gauss_k: int = 3
    clahe_c: float = 2.0
    clahe_g: Tuple[int, int] = (8, 8)
    med_k: int = 9
    thresh_adaptive_method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh_type: int = cv2.THRESH_BINARY_INV  # Use INV if dots are dark on light bg
    thresh_block_size: int = 37
    thresh_c: int = 7
    morph_open_k: int = 3
    morph_open_iterations: int = 1


@dataclass
class BlobDetectorParams:
    minThreshold: float = 10
    maxThreshold: float = 220
    thresholdStep: float = 10
    minRepeatability: int = 2
    minDistBetweenBlobs: float = 10  # Adjust based on expected dot spacing

    # Filters
    filterByColor: bool = True
    blobColor: int = 0  # 0 for dark blobs, 255 for light blobs

    filterByArea: bool = True
    minArea: float = 30  # Adjust based on expected dot size range
    maxArea: float = 500  # Adjust based on expected dot size range

    filterByCircularity: bool = True
    minCircularity: float = 0.7  # Keep high for circles
    maxCircularity: float = 1.0

    filterByInertia: bool = True
    minInertiaRatio: float = 0.4  # Lower values allow more elongation (perspective)
    maxInertiaRatio: float = 1.0

    filterByConvexity: bool = True
    minConvexity: float = 0.85  # Keep high for convex shapes
    maxConvexity: float = 1.0


@dataclass
class RefinementParams:
    win_size: Tuple[int, int] = (5, 5)
    zero_zone: Tuple[int, int] = (-1, -1)
    criteria: Tuple[int, int, float] = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        40,
        0.001,
    )


@dataclass
class GridDetectionConfig:
    preprocessing: PreprocessingParams = field(default_factory=PreprocessingParams)
    blob_detector: BlobDetectorParams = field(default_factory=BlobDetectorParams)
    refinement: RefinementParams = field(default_factory=RefinementParams)
    # RANSAC/Sorting params will be added later


# --- Feature Extractor Class ---
class FeatureExtractorCV:
    def __init__(self, config: Optional[GridDetectionConfig] = None):
        self.image_folder_path = None
        self.ref_images_dict = {}
        self.calib_images_dict = {}
        self.config = config if config else GridDetectionConfig()
        self.detected_grids = {}  # Store results per folder

    def load_images(self, image_folder_path: Path):
        """
        Load reference and calibration images from folder structure.

        Args:
            image_folder_path: Path to folders containing images

        Returns:
            Tuple of (ref_images_dict, calib_images_dict) where:
            - ref_images_dict: Dictionary mapping folder indices to reference images
            - calib_images_dict: Dictionary mapping folder indices to lists of calibration images
        """
        self.image_folder_path = image_folder_path

        folders = sorted(
            self.image_folder_path.iterdir(),
            key=lambda x: int(x.stem) if x.stem.isdigit() else -1,
        )

        for folder in folders:
            folder_stem = folder.stem
            folder_idx = int(folder_stem)

            img_files = sorted(
                folder.iterdir(),
                key=lambda x: int(x.stem) if x.stem.isdigit() else -1,
            )

            folder_ref_image = None
            folder_calib_images = []

            for img in img_files:
                if img.name == "0.npy":
                    folder_ref_image = ImageProcessor.load_image(img)
                elif img.suffix == ".npy":
                    folder_calib_images.append(ImageProcessor.load_image(img))

            if folder_ref_image is None or not folder_calib_images:
                continue

            self.ref_images_dict[folder_idx] = folder_ref_image
            self.calib_images_dict[folder_idx] = folder_calib_images

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Applies preprocessing steps based on config."""
        if image.dtype != np.uint8:
            if np.max(image) <= 1.0 and np.min(image) >= 0.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)

        if len(image.shape) == 3:
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            processed = image.copy()

        cfg = self.config.preprocessing

        if cfg.gauss_k > 0:
            processed = cv2.GaussianBlur(processed, (cfg.gauss_k, cfg.gauss_k), 0)

        if cfg.clahe_c > 0:
            clahe = cv2.createCLAHE(clipLimit=cfg.clahe_c, tileGridSize=cfg.clahe_g)
            processed = clahe.apply(processed)

        if cfg.med_k > 0:
            processed = cv2.medianBlur(processed, cfg.med_k)

        bs = cfg.thresh_block_size
        if bs <= 1:
            bs = 3
        elif bs % 2 == 0:
            bs += 1

        binary_img = cv2.adaptiveThreshold(
            processed,
            255,
            cfg.thresh_adaptive_method,
            cfg.thresh_type,
            bs,
            cfg.thresh_c,
        )

        if cfg.morph_open_k > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (cfg.morph_open_k, cfg.morph_open_k)
            )
            binary_img = cv2.morphologyEx(
                binary_img, cv2.MORPH_OPEN, kernel, iterations=cfg.morph_open_iterations
            )

        return (
            processed  # Return grayscale image processed by CLAHE/blur for refinement
        )

    def _detect_candidates(self, image: np.ndarray) -> np.ndarray:
        """Detects candidate blob centers using SimpleBlobDetector."""
        cfg = self.config.blob_detector
        params = cv2.SimpleBlobDetector_Params()

        params.minThreshold = cfg.minThreshold
        params.maxThreshold = cfg.maxThreshold
        params.thresholdStep = cfg.thresholdStep
        params.minRepeatability = cfg.minRepeatability
        params.minDistBetweenBlobs = cfg.minDistBetweenBlobs

        params.filterByColor = cfg.filterByColor
        params.blobColor = cfg.blobColor

        params.filterByArea = cfg.filterByArea
        params.minArea = cfg.minArea
        params.maxArea = cfg.maxArea

        params.filterByCircularity = cfg.filterByCircularity
        params.minCircularity = cfg.minCircularity
        params.maxCircularity = cfg.maxCircularity

        params.filterByInertia = cfg.filterByInertia
        params.minInertiaRatio = cfg.minInertiaRatio
        params.maxInertiaRatio = cfg.maxInertiaRatio

        params.filterByConvexity = cfg.filterByConvexity
        params.minConvexity = cfg.minConvexity
        params.maxConvexity = cfg.maxConvexity

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(image)

        if not keypoints:
            return np.array([], dtype=np.float32).reshape(0, 2)

        points = cv2.KeyPoint_convert(keypoints)
        return points.astype(np.float32)

    def _refine_candidates(
        self, image: np.ndarray, candidates: np.ndarray
    ) -> np.ndarray:
        """Refines candidate points to subpixel accuracy."""
        if candidates.shape[0] == 0:
            return candidates

        cfg = self.config.refinement
        if image.dtype != np.uint8:
            if np.max(image) <= 1.0 and np.min(image) >= 0.0:
                image_gray = (image * 255).astype(np.uint8)
            else:
                image_gray = np.clip(image, 0, 255).astype(np.uint8)
            if len(image_gray.shape) == 3:
                image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image

        refined_points = cv2.cornerSubPix(
            image_gray, candidates, cfg.win_size, cfg.zero_zone, cfg.criteria
        )
        return refined_points

    def detect_grid_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detects the dot grid in a single image.

        Args:
            image: Input grayscale image (np.ndarray uint8).

        Returns:
            Sorted grid points (Nx2 np.ndarray float32) or None if detection fails.
        """
        try:
            processed_image = self._preprocess_image(image)

            candidates = self._detect_candidates(processed_image)
            if candidates.shape[0] < 4:  # Need at least a few points for grid fitting
                print(
                    f"Warning: Found only {candidates.shape[0]} candidates. Skipping grid fitting."
                )
                return None

            refined_candidates = self._refine_candidates(processed_image, candidates)

            final_grid_points = refined_candidates  # TEMPORARY

            if final_grid_points is None or final_grid_points.shape[0] == 0:
                print("Warning: Grid fitting failed.")
                return None

            return final_grid_points

        except Exception as e:
            print(f"Error during grid detection: {e}")
            return None

    def process_all_images(self):
        """Processes all loaded calibration images to detect grid features."""
        print("Starting grid detection for all loaded images...")
        self.detected_grids = {}

        for folder_idx, calib_images in self.calib_images_dict.items():
            print(f"Processing folder {folder_idx}...")
            folder_grids = []
            for i, img in enumerate(calib_images):
                print(f"  Processing image {i + 1}/{len(calib_images)}...")
                grid_points = self.detect_grid_features(img)
                if grid_points is not None:
                    folder_grids.append(grid_points)
                    print(f"    Detected {len(grid_points)} points.")
                else:
                    print(f"    Detection failed for image {i + 1}.")
                    folder_grids.append(None)  # Keep placeholder for failed images

            self.detected_grids[folder_idx] = folder_grids
            print(
                f"Finished folder {folder_idx}. Found grids in {sum(g is not None for g in folder_grids)}/{len(folder_grids)} images."
            )


class FeatureAlignerCV:
    def __init__(self, grid_spacing: float = 1.0):
        """
        Initialize the feature aligner.

        Args:
            grid_spacing: Physical spacing between grid points in world units (e.g., mm).
        """
        self.grid_spacing = grid_spacing
        # self.aligned_features = {} # No longer used directly
        # self.correspondences = {} # No longer used directly
        self.processed_results = {}  # Store final processed results

    def find_center_point(self, points: np.ndarray) -> Optional[np.ndarray]:
        """
        Find a point near the center of the grid to use as reference origin.

        Args:
            points: Array of detected grid points (Nx2) in image coordinates.

        Returns:
            The selected center point (in image coordinates), or None if input is invalid.
        """
        if points is None or len(points) == 0:
            return None
        if points.ndim != 2 or points.shape[1] != 2:
            print(
                f"Warning: Invalid shape for points in find_center_point: {points.shape}"
            )
            return None

        # Find geometric center of all points
        center = np.mean(points, axis=0)

        # Find the point closest to the geometric center
        distances = np.linalg.norm(points - center, axis=1)
        center_idx = np.argmin(distances)

        return points[center_idx]

    def _estimate_basis_vectors(
        self,
        points: np.ndarray,
        ref_point: np.ndarray,
        initial_pixel_spacing: float,
        num_neighbors: int = 8,
        angle_tolerance: float = 30.0,  # Degrees tolerance for grouping vectors
        dist_tolerance_factor: float = 0.5,  # Tolerance for neighbor distance relative to initial_pixel_spacing
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Estimates the grid basis vectors (vec_col, vec_row) near the reference point.

        Args:
            points: All detected points (Nx2).
            ref_point: The reference point (1x2 or 2,).
            initial_pixel_spacing: An initial estimate of spacing for distance filtering.
            num_neighbors: How many nearest neighbors to consider.
            angle_tolerance: Tolerance in degrees for grouping vectors by angle.
            dist_tolerance_factor: Multiplier for initial_pixel_spacing to define distance tolerance.

        Returns:
            Tuple (vec_col, vec_row) representing grid steps, or None if failed.
        """
        if points is None or len(points) < num_neighbors + 1 or ref_point is None:
            print("Warning: Not enough points to estimate basis vectors.")
            return None

        ref_point_flat = ref_point.flatten()
        if ref_point_flat.shape[0] != 2:
            print(f"Warning: Invalid ref_point shape {ref_point.shape}")
            return None

        # Use KDTree for efficient nearest neighbor search
        try:
            kdtree = KDTree(points)
            # Query for num_neighbors + 1 because the point itself will be the closest
            distances, indices = kdtree.query(ref_point_flat, k=num_neighbors + 1)
        except Exception as e:
            print(f"Error during KDTree query: {e}")
            return None

        # Filter neighbors based on distance to ref_point
        neighbor_vectors = []
        min_dist = initial_pixel_spacing * (1.0 - dist_tolerance_factor)
        max_dist = initial_pixel_spacing * (1.0 + dist_tolerance_factor)

        # Start from 1 to skip the reference point itself (distance=0)
        for i in range(1, len(indices)):
            neighbor_idx = indices[i]
            dist = distances[i]

            # Check if distance is within expected range
            if min_dist <= dist <= max_dist:
                neighbor_point = points[neighbor_idx]
                vec = neighbor_point - ref_point_flat
                neighbor_vectors.append(vec)

        if len(neighbor_vectors) < 2:
            print(
                f"Warning: Found only {len(neighbor_vectors)} potential neighbors within distance tolerance. Cannot estimate basis vectors."
            )
            return None

        # Group vectors by angle
        angles = (
            np.arctan2(
                np.array(neighbor_vectors)[:, 1], np.array(neighbor_vectors)[:, 0]
            )
            * 180
            / np.pi
        )
        vectors_by_angle = {}  # Map angle_group -> list of vectors

        # Simple angle grouping (e.g., near 0, 90, 180, 270 degrees)
        # Normalize angles to [0, 360)
        angles_norm = (angles % 360 + 360) % 360

        # Group vectors: Iterate and merge similar angles
        angle_groups = []  # List of lists of indices
        visited = [False] * len(angles_norm)
        for i in range(len(angles_norm)):
            if visited[i]:
                continue
            current_group_indices = [i]
            visited[i] = True
            for j in range(i + 1, len(angles_norm)):
                if not visited[j]:
                    # Calculate angular difference carefully (handle wrap-around)
                    diff = abs(angles_norm[i] - angles_norm[j])
                    angular_diff = min(diff, 360 - diff)
                    if angular_diff <= angle_tolerance:
                        current_group_indices.append(j)
                        visited[j] = (
                            True  # Mark as visited to avoid adding to multiple groups
                        )
            angle_groups.append(current_group_indices)

        if len(angle_groups) < 2:
            print(
                f"Warning: Could not form at least two distinct angle groups for basis vectors. Found {len(angle_groups)} groups."
            )
            return None

        # Calculate average vector for each group
        avg_vectors = []
        for group_indices in angle_groups:
            if len(group_indices) > 0:
                group_vecs = np.array([neighbor_vectors[idx] for idx in group_indices])
                avg_vec = np.mean(group_vecs, axis=0)
                avg_vectors.append(avg_vec)

        # Select the two most orthogonal vectors closest to expected length
        best_pair = None
        min_orthogonality_diff = float("inf")
        target_length = initial_pixel_spacing

        for i in range(len(avg_vectors)):
            for j in range(i + 1, len(avg_vectors)):
                v1 = avg_vectors[i]
                v2 = avg_vectors[j]
                len1 = np.linalg.norm(v1)
                len2 = np.linalg.norm(v2)

                # Check length similarity to target
                if not (min_dist <= len1 <= max_dist and min_dist <= len2 <= max_dist):
                    continue

                # Check orthogonality (dot product close to zero)
                cos_theta = np.dot(v1, v2) / (len1 * len2)
                ortho_diff = abs(abs(cos_theta) - 0)  # Deviation from cos(90deg)=0

                # Prefer pairs closer to orthogonal and closer to target length
                score = (
                    ortho_diff
                    + 0.1
                    * (abs(len1 - target_length) + abs(len2 - target_length))
                    / target_length
                )

                if score < min_orthogonality_diff:
                    min_orthogonality_diff = score
                    # Ensure consistent orientation if possible (e.g., v_col positive x, v_row positive y)
                    angle1 = np.arctan2(v1[1], v1[0]) * 180 / np.pi
                    angle2 = np.arctan2(v2[1], v2[0]) * 180 / np.pi
                    # Basic check: one near horizontal, one near vertical
                    # More robust: check relative angle is near +/- 90 deg
                    relative_angle_diff = abs(angle1 - angle2)
                    relative_angle = min(relative_angle_diff, 360 - relative_angle_diff)

                    if (
                        abs(relative_angle - 90) < angle_tolerance * 2
                    ):  # Allow larger tolerance for relative angle
                        # Assign based on angle (heuristic: smaller angle is col)
                        if (
                            abs(angle1) < 45 or abs(angle1 - 180) < 45
                        ):  # Closer to horizontal
                            best_pair = (
                                (v1, v2)
                                if abs(angle2 - 90) < 45 or abs(angle2 - 270) < 45
                                else None
                            )
                        elif (
                            abs(angle2) < 45 or abs(angle2 - 180) < 45
                        ):  # Closer to horizontal
                            best_pair = (
                                (v2, v1)
                                if abs(angle1 - 90) < 45 or abs(angle1 - 270) < 45
                                else None
                            )

        if best_pair is None:
            print(
                "Warning: Could not find two suitable orthogonal basis vectors among neighbors."
            )
            # Fallback or more sophisticated selection could be added here
            # For now, just select the two vectors with length closest to pixel_spacing
            if len(avg_vectors) >= 2:
                avg_vectors.sort(
                    key=lambda v: abs(np.linalg.norm(v) - initial_pixel_spacing)
                )
                print(
                    "Warning: Falling back to two closest length vectors (may not be orthogonal)."
                )
                best_pair = (
                    avg_vectors[0],
                    avg_vectors[1],
                )  # Arbitrary assignment to col/row here
                # Try to guess orientation
                angle0 = np.arctan2(best_pair[0][1], best_pair[0][0]) * 180 / np.pi
                angle1 = np.arctan2(best_pair[1][1], best_pair[1][0]) * 180 / np.pi
                if abs(angle0) < 45 or abs(angle0 - 180) < 45:  # vec0 is likely col
                    vec_col = best_pair[0]
                    vec_row = best_pair[1]
                elif abs(angle1) < 45 or abs(angle1 - 180) < 45:  # vec1 is likely col
                    vec_col = best_pair[1]
                    vec_row = best_pair[0]
                else:  # Default guess
                    vec_col, vec_row = best_pair

            else:
                return None

        # Ensure positive orientation if possible (heuristic)
        if best_pair[0][0] < 0:  # Prioritize vec_col having positive x component
            vec_col = -best_pair[0]
        else:
            vec_col = best_pair[0]
        if best_pair[1][1] < 0:  # Prioritize vec_row having positive y component
            vec_row = -best_pair[1]
        else:
            vec_row = best_pair[1]

        # Final check: ensure vectors are not collinear
        cosine_similarity = np.dot(vec_col, vec_row) / (
            np.linalg.norm(vec_col) * np.linalg.norm(vec_row)
        )
        if (
            abs(cosine_similarity) > 0.95
        ):  # Check if vectors are too close to parallel/anti-parallel
            print(
                f"Warning: Estimated basis vectors are nearly collinear (cosine similarity: {cosine_similarity:.3f})."
            )
            return None

        return vec_col, vec_row

    def create_correspondences(
        self, original_points: np.ndarray, ref_point: np.ndarray
    ) -> Tuple[
        Optional[Dict[Tuple[int, int], np.ndarray]],
        Optional[Dict[Tuple[int, int], np.ndarray]],
        Optional[np.ndarray],  # Basis Matrix B
        Optional[np.ndarray],  # Inverse Basis Matrix B_inv
    ]:
        """
        Create 2D-3D correspondences using estimated basis vectors.

        Args:
            original_points: Array of detected grid points (Nx2) in image coordinates.
            ref_point: The reference point (origin) in image coordinates.

        Returns:
            Tuple of (image_coord_map, world_coord_map, B, B_inv), or (None, None, None, None) if failed:
            - image_coord_map: Dictionary mapping (row, col) -> 2D image point.
            - world_coord_map: Dictionary mapping (row, col) -> 3D world point.
            - B: Basis matrix [vec_col | vec_row].
            - B_inv: Inverse of basis matrix.
        """
        if (
            original_points is None or len(original_points) < 4 or ref_point is None
        ):  # Need more points for robust basis estimation
            print(
                "Warning: Insufficient points or no reference point for correspondence creation."
            )
            return None, None, None, None

        ref_point_flat = ref_point.flatten()

        # --- Estimate initial pixel spacing (heuristic for neighbor filtering) ---
        relative_points_for_spacing = original_points - ref_point_flat
        dist_matrix = np.sqrt(
            np.sum(
                (
                    relative_points_for_spacing[:, np.newaxis, :]
                    - relative_points_for_spacing[np.newaxis, :, :]
                )
                ** 2,
                axis=2,
            )
        )
        min_dists = []
        for i in range(len(dist_matrix)):
            row = dist_matrix[i]
            row_filtered = row[row > 1e-6]
            if len(row_filtered) > 0:
                min_dists.append(np.min(row_filtered))
        if not min_dists:
            print("Warning: Could not estimate initial pixel spacing.")
            return None, None, None, None
        initial_pixel_spacing = np.median(min_dists)
        if initial_pixel_spacing < 1e-6:
            print(
                f"Warning: Estimated initial pixel spacing is near zero ({initial_pixel_spacing})."
            )
            return None, None, None, None
        # --- End of initial spacing estimation ---

        # --- Estimate Basis Vectors ---
        basis_vectors = self._estimate_basis_vectors(
            original_points, ref_point_flat, initial_pixel_spacing
        )
        if basis_vectors is None:
            print("Warning: Failed to estimate basis vectors.")
            return None, None, None, None
        vec_col, vec_row = basis_vectors
        # --- End of Basis Vector Estimation ---

        # --- Create Basis Matrix and Inverse ---
        # B = [vec_col, vec_row] (vectors as columns)
        B = np.array([vec_col, vec_row]).T
        try:
            # Check condition number?
            if np.linalg.det(B) == 0:
                raise np.linalg.LinAlgError("Basis matrix is singular.")
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError as e:
            print(f"Error inverting basis matrix B: {e}. B=\n{B}")
            return None, None, None, None
        # --- End of Matrix Creation ---

        # --- Assign grid coordinates using basis vectors ---
        image_coord_map = {}
        world_coord_map = {}
        assigned_coords: Set[Tuple[int, int]] = set()
        calculated_indices = {}  # Store float indices before rounding for collision check

        # Find the index of the ref_point in original_points
        ref_point_index = -1
        min_ref_dist = float("inf")
        for i, p in enumerate(original_points):
            dist = np.linalg.norm(p - ref_point_flat)
            if dist < 1e-6:  # Use tolerance for floating point comparison
                ref_point_index = i
                break  # Found exact match

        if ref_point_index == -1:
            # If no exact match, find the closest one (should have been found by find_center_point)
            distances = np.linalg.norm(original_points - ref_point_flat, axis=1)
            ref_point_index = np.argmin(distances)
            if (
                np.min(distances) > 1.0
            ):  # If even the closest is far, something is wrong
                print(
                    f"Warning: Could not reliably find the reference point index. Closest distance: {np.min(distances)}"
                )
                return None, None, None, None

        # Assign grid coordinates based on projection
        for i, point in enumerate(original_points):
            relative_point = point - ref_point_flat

            # Project onto basis: indices_float = B_inv * relative_point
            indices_float = B_inv @ relative_point
            col_f = indices_float[0]
            row_f = indices_float[1]

            col = int(round(col_f))
            row = int(round(row_f))
            grid_coord = (row, col)

            # Check if this is the reference point index. It must map to (0,0).
            if i == ref_point_index:
                if grid_coord != (0, 0):
                    print(
                        f"Info: Reference point index {i} ({point}) did not map exactly to (0,0) initially (calc: {grid_coord}, float: ({col_f:.2f}, {row_f:.2f})). Forcing to (0,0)."
                    )
                    # Potential issue if basis vectors are poorly estimated.
                grid_coord = (0, 0)
                row, col = 0, 0

            # Handle potential collisions: prioritize point closest to its projected integer grid location
            # Ideal projected location in relative coords: ideal_rel = col * vec_col + row * vec_row
            ideal_relative_location = B @ np.array([col, row])
            current_dist_from_ideal = np.linalg.norm(
                relative_point - ideal_relative_location
            )

            if grid_coord in assigned_coords:
                existing_point_idx = calculated_indices[grid_coord]["index"]
                existing_dist = calculated_indices[grid_coord]["dist_from_ideal"]

                if current_dist_from_ideal < existing_dist:
                    # Current point is a better fit for this grid coord, overwrite
                    # (Need to remove the old point from the map later or handle carefully)
                    # print(f"Collision at {grid_coord}: New point {i} closer ({current_dist_from_ideal:.2f}) than old point {existing_point_idx} ({existing_dist:.2f}). Replacing.")
                    # Mark old point for removal? For now, allow overwrite in maps below.
                    pass  # Allow overwrite
                else:
                    # Existing point was closer, keep it and skip current point
                    # print(f"Collision at {grid_coord}: Keeping old point {existing_point_idx} ({existing_dist:.2f}). Skipping new point {i} ({current_dist_from_ideal:.2f}).")
                    continue  # Skip assignment below

            # Store ORIGINAL image point and calculated world point
            image_coord_map[grid_coord] = point
            world_coord_map[grid_coord] = np.array(
                [col * self.grid_spacing, row * self.grid_spacing, 0.0],
                dtype=np.float32,
            )
            assigned_coords.add(grid_coord)
            calculated_indices[grid_coord] = {
                "index": i,
                "dist_from_ideal": current_dist_from_ideal,
            }  # Store info for collision checks

        # Final check: Ensure reference point is (0,0)
        if (0, 0) not in image_coord_map or not np.allclose(
            image_coord_map[(0, 0)], original_points[ref_point_index], atol=1e-3
        ):
            print(
                f"Warning: Final check failed. Reference point {original_points[ref_point_index]} is not mapped to (0,0) in final map. Map[(0,0)] = {image_coord_map.get((0, 0))}"
            )
            # Consider returning None if this happens

        if len(image_coord_map) < 4:
            print(f"Warning: Only {len(image_coord_map)} points mapped successfully.")
            # Optionally return None

        return image_coord_map, world_coord_map, B, B_inv
        # --- End of coordinate assignment ---

    def process_grid_points(
        self, grid_points_dict: Dict[int, List[Optional[np.ndarray]]]
    ) -> Dict[int, List[Optional[Dict[str, Any]]]]:
        """
        Process all detected grid points to create 2D-3D correspondences.
        Uses improved basis vector estimation.

        Args:
            grid_points_dict: Dictionary mapping folder index to a list of detected
                              grid points (Nx2 np.ndarray) for each image.
                              List items can be None if detection failed for an image.

        Returns:
            Dictionary mapping folder index to a list of processing results for each image.
            Each result is a dictionary containing:
            - 'image_points': (Mx2 np.ndarray) Detected 2D points in image coordinates.
            - 'world_points': (Mx3 np.ndarray) Corresponding 3D points in world coordinates.
            - 'reference_point': (1x2 np.ndarray) The 2D image point chosen as the reference (origin).
            - 'grid_indices': List of (row, col) tuples corresponding to the points.
            - 'image_coord_map': Dict mapping (row, col) -> 2D image point.
            - 'world_coord_map': Dict mapping (row, col) -> 3D world point.
            - 'basis_matrix': Estimated basis matrix B = [vec_col | vec_row].
            - 'basis_matrix_inv': Inverse of B.
            Result is None if processing failed for an image.
        """
        self.processed_results = {}  # Clear previous results

        for folder_idx, folder_grids in grid_points_dict.items():
            folder_results = []

            for i, original_grid_points in enumerate(folder_grids):
                # Need at least a few points for robust basis estimation
                if (
                    original_grid_points is None or len(original_grid_points) < 8
                ):  # Increased minimum points
                    print(
                        f"Info: Skipping folder {folder_idx}, image {i} due to insufficient points ({len(original_grid_points) if original_grid_points is not None else 0}). Need at least 8."
                    )
                    folder_results.append(None)
                    continue

                # Find center point to use as origin (using original points)
                ref_point = self.find_center_point(original_grid_points)
                if ref_point is None:
                    print(
                        f"Warning: Could not find reference point for folder {folder_idx}, image {i}. Skipping."
                    )
                    folder_results.append(None)
                    continue

                # Create correspondences using original points and the reference point
                image_coord_map, world_coord_map, basis_matrix, basis_matrix_inv = (
                    self.create_correspondences(original_grid_points, ref_point)
                )

                # Check if correspondence creation failed
                if (
                    image_coord_map is None
                    or world_coord_map is None
                    or not image_coord_map
                    or basis_matrix is None
                    or basis_matrix_inv is None
                ):
                    print(
                        f"Warning: Failed to create correspondences for folder {folder_idx}, image {i}. Skipping."
                    )
                    folder_results.append(None)
                    continue

                # Create ordered lists of corresponding points and indices for easy use
                try:
                    grid_indices = sorted(image_coord_map.keys())
                except TypeError as e:
                    print(
                        f"Error sorting grid indices: {e}. Keys: {list(image_coord_map.keys())}"
                    )
                    folder_results.append(None)
                    continue

                image_points_list = [image_coord_map[idx] for idx in grid_indices]
                world_points_list = [world_coord_map[idx] for idx in grid_indices]

                # Final check for consistency
                if len(image_points_list) != len(world_points_list):
                    print(
                        f"Error: Mismatch in length between image points ({len(image_points_list)}) and world points ({len(world_points_list)}) for folder {folder_idx}, image {i}."
                    )
                    folder_results.append(None)
                    continue
                if len(image_points_list) == 0:
                    print(
                        f"Warning: No points were successfully mapped for folder {folder_idx}, image {i}."
                    )
                    folder_results.append(None)
                    continue

                result = {
                    "image_points": np.array(image_points_list, dtype=np.float32),
                    "world_points": np.array(world_points_list, dtype=np.float32),
                    "reference_point": ref_point.astype(np.float32),
                    "grid_indices": grid_indices,
                    "image_coord_map": image_coord_map,
                    "world_coord_map": world_coord_map,
                    "basis_matrix": basis_matrix.astype(np.float32),
                    "basis_matrix_inv": basis_matrix_inv.astype(np.float32),
                }

                folder_results.append(result)

            self.processed_results[folder_idx] = folder_results

        return self.processed_results
