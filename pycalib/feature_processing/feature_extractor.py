import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class ImageProcessParams:
    """Parameters for image processing"""

    gauss_k: int = 3  # For cv2.GaussianBlur kernel size
    med_k: int = 9  # For cv2.medianBlur kernel
    clahe_c: float = 2.0  # For cv2.createCLAHE clipLimit
    clahe_g: Tuple[int, int] = (8, 8)  # For cv2.createCLAHE tileGridSize
    thresh_b: int = 37  # For cv2.adaptiveThreshold blockSize
    thresh_c: int = 7  # For cv2.adaptiveThreshold C value
    morph_k: int = 9  # For morphology kernel
    morph_i: int = 1  # For morphologyEx iterations


@dataclass
class FeatureAlignParams:
    """Parameters for feature alignment to grid"""

    grid_spacing_mm = 0.5
    frame_spacing_mm = 0.25

    neighbor_count: int = 8
    vec_round_thresh: float = 0.0
    max_iterations: int = 20


class FeatureDetectorParams:
    """Parameters for feature detection"""

    min_area: int = 50
    max_area: int = 750
    circular_thresh: float = 0.75


class ImageProcessor:
    """Image processing utilities for feature detection.

    Handles image preprocessing, vignetting correction, and grid analysis.

    Args:
        image: Input image as uint8 numpy array
        params: Optional processing parameters
    """

    def __init__(
        self, image: np.ndarray, params: Optional[ImageProcessParams] = None
    ) -> None:
        self._raw_image = image.astype(np.uint8).copy()
        self._params = params or ImageProcessParams()

    def process_image(self):
        from skimage.draw import disk

        if np.max(self._raw_image) <= 1.0:
            self._raw_image *= 255
            self._raw_image = self._raw_image.astype(np.uint8)

        img = cv2.GaussianBlur(
            self._raw_image,
            (self._params.gauss_k, self._params.gauss_k),
            0,
        )

        clahe = cv2.createCLAHE(
            clipLimit=self._params.clahe_c,
            tileGridSize=self._params.clahe_g,
        )
        img_clahe = clahe.apply(img)

        img_blurred = cv2.medianBlur(img_clahe, self._params.med_k)

        img_binary = cv2.adaptiveThreshold(
            img_blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self._params.thresh_b,
            self._params.thresh_c,
        )

        ks = self._params.morph_k
        kernel = np.ones((ks, ks), dtype=np.uint8)
        rr, cc = disk((int(ks / 2), int(ks / 2)), int(ks / 2))
        kernel[rr, cc] = 1

        img_binary = cv2.morphologyEx(
            img_binary,
            cv2.MORPH_OPEN,
            kernel,
            iterations=self._params.morph_i,
        )

        return img_binary

    @staticmethod
    def load_image(image_path: Union[str, Path]) -> np.ndarray[np.uint8]:
        """Load and preprocess an image from file.

        Args:
            image_path: Path to image file (.jpg, .png, .npy)

        Returns:
            Image as uint8 numpy array

        Raises:
            ValueError: If image loading fails
            Exception: For other loading/processing errors
        """
        try:
            path = Path(image_path)
            if path.suffix == ".npy":
                image = np.load(str(path)).copy()
                if np.max(image) <= 1.0:
                    image = (image * 255).astype(np.uint8)
            else:
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"Failed to load image: {path}")

            return image.astype(np.uint8)
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")


class FeatureDetector:
    def __init__(
        self, image: np.ndarray, params: Optional[FeatureDetectorParams] = None
    ) -> None:
        if len(image.shape) != 2:
            raise ValueError("Input image must be 2D grayscale")
        self._image = image
        self._params = params or FeatureDetectorParams()

    def detect(self) -> np.ndarray:
        """Detect circular features in the image.

        Returns:
            np.ndarray: Array of feature centroids (Nx2)

        Raises:
            ValueError: If no contours are found
        """
        try:
            contours = cv2.findContours(
                self._image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )[0]

            if not contours:
                return np.array([])

            img_points = []
            for contour in contours:
                m = cv2.moments(contour)
                if m["m00"] <= 0:
                    continue

                cx = m["m10"] / m["m00"]
                cy = m["m01"] / m["m00"]
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)

                circularity = 0
                if perimeter > 0:
                    circularity = 4 * math.pi * area / (perimeter**2)

                if (
                    self._params.min_area < area < self._params.max_area
                    and circularity > self._params.circular_thresh
                ):
                    img_points.append([cx, cy])

            return np.asarray(img_points)

        except Exception as e:
            raise ValueError(f"Error detecting features: {str(e)}")


class FeatureAligner:
    """Align detected features to a regular grid pattern.

    Args:
        FeatureAlignParams
    """

    def __init__(self, params: Optional[FeatureAlignParams] = None) -> None:
        self._params = params or FeatureAlignParams()
        self._dist_thresh_factor = 1.3
        self._min_dist_ratio_thresh = 1.1
        self._max_dist_ratio_thresh = 1.7
        self._min_points_for_thresh_estimation = 10
        self._default_dist_ratio_thresh = 1.3

    def _estimate_distance_threshold(self, centroids: np.ndarray) -> float:
        """
        Estimates the distance ratio threshold dynamically based on the
        median nearest neighbor distance.

        Args:
            centroids: Array of detected feature centroids (Nx2).

        Returns:
            The estimated distance ratio threshold.
        """
        if len(centroids) < self._min_points_for_thresh_estimation:
            return self._default_dist_ratio_thresh

        try:
            nbrs = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(centroids)
            distances, _ = nbrs.kneighbors(centroids)

            nn_distances = distances[:, 1]

            nn_distances = nn_distances[nn_distances > 1e-6]

            if len(nn_distances) < self._min_points_for_thresh_estimation // 2:
                return self._default_dist_ratio_thresh

            median_nn_dist = np.median(nn_distances)

            if median_nn_dist < 1e-6:
                return self._default_dist_ratio_thresh

            estimated_threshold = self._dist_thresh_factor

            clamped_threshold = np.clip(
                estimated_threshold,
                self._min_dist_ratio_thresh,
                self._max_dist_ratio_thresh,
            )

            return clamped_threshold

        except Exception as e:
            print(
                f"Error estimating distance threshold: {e}. Using default: {self._default_dist_ratio_thresh}"
            )
            return self._default_dist_ratio_thresh

    def _align_points_to_grid(
        self,
        centroids: np.ndarray,
        tracked_ref_point: np.ndarray,
        frame_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """Align detected feature points relative to the tracked reference point.

        Args:
            centroids: Centroid coordinates of all detected features in the frame.
            tracked_ref_point: The specific centroid identified as the origin/reference.
            frame_idx: Index of the current frame.

        Returns:
            img_points: Array of shape (N,2) containing valid, aligned feature coordinates.
            obj_points: Array of shape (N,3) containing corresponding 3D object coordinates.
            correspondences: A list of dictionaries detailing each aligned point pair.
                             Each dict contains: {'id': int, 'img_point': ndarray(2,),
                             'obj_point': ndarray(3,), 'is_reference': bool}.
        """
        if (
            not isinstance(centroids, np.ndarray)
            or centroids.ndim != 2
            or centroids.shape[1] != 2
        ):
            raise ValueError("centroids must be an Nx2 numpy array.")
        if tracked_ref_point is None:
            raise ValueError("tracked_ref_point cannot be None.")
        if tracked_ref_point.shape != (2,):
            raise ValueError("tracked_ref_point must have shape (2,).")
        if len(centroids) == 0:
            print(
                f"Warning: centroids array is empty for frame {frame_idx}. Returning empty points."
            )
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32), []

        distances = np.linalg.norm(centroids - tracked_ref_point, axis=1)
        origin_idx = np.argmin(distances)
        min_dist = distances[origin_idx]
        if min_dist > 0.5:
            print(
                f"Warning: Tracked reference point {tracked_ref_point} "
                f"is too far ({min_dist:.4f} pixels) from the closest detected centroid "
                f"in frame {frame_idx}. Alignment aborted for this frame."
            )
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32), []

        distance_ratio_thresh = self._estimate_distance_threshold(centroids)

        obj_points = np.empty((centroids.shape[0], 3), dtype=float)
        obj_points.fill(np.nan)
        obj_points[origin_idx] = [0.0, 0.0, frame_idx * self._params.frame_spacing_mm]

        iterations = 0
        assigned_in_last_iter = True
        total_points = len(centroids)

        while (
            np.isnan(obj_points[:, :2]).any()
            and iterations < self._params.max_iterations
            and assigned_in_last_iter
        ):
            assigned_in_last_iter = False
            assigned_mask = ~np.isnan(obj_points[:, 0])
            assigned_indices = np.where(assigned_mask)[0]

            for i in assigned_indices:
                current_centroid = centroids[i]
                current_obj_pt = obj_points[i]

                k_neighbors = min(self._params.neighbor_count + 1, len(centroids))
                if k_neighbors <= 1:
                    continue

                dists, neighbor_indices = self.get_dists(
                    centroids, current_centroid, k=k_neighbors
                )

                valid_neighbors_mask = neighbor_indices != i
                neighbor_indices = neighbor_indices[valid_neighbors_mask]
                dists = dists[valid_neighbors_mask]

                if not len(neighbor_indices):
                    continue

                min_dist_nn = np.min(dists)
                if min_dist_nn < 1e-9:
                    continue

                neighbor_centroids = centroids[neighbor_indices]
                vecs = neighbor_centroids - current_centroid
                norms = np.linalg.norm(vecs, axis=1)

                for j, neighbor_idx in enumerate(neighbor_indices):
                    if np.isnan(obj_points[neighbor_idx, 0]):
                        dist_ratio = dists[j] / min_dist_nn
                        if dist_ratio < distance_ratio_thresh:
                            if norms[j] < 1e-9:
                                continue
                            unit_vec = vecs[j] / norms[j]
                            rounded_vec = np.round(unit_vec, 0)
                            passes_vec_round_check = (
                                np.abs(rounded_vec[0] * rounded_vec[1])
                                <= self._params.vec_round_thresh
                            )

                            if passes_vec_round_check:
                                grid_step = rounded_vec * self._params.grid_spacing_mm
                                if np.all(np.isclose(grid_step, 0)):
                                    continue

                                new_xy = current_obj_pt[:2] + grid_step
                                obj_points[neighbor_idx] = [
                                    new_xy[0],
                                    new_xy[1],
                                    frame_idx * self._params.frame_spacing_mm,
                                ]
                                assigned_in_last_iter = True

            num_assigned = np.sum(~np.isnan(obj_points[:, 0]))
            if total_points > 0 and (num_assigned / total_points >= 0.95):
                break

            iterations += 1

        valid_mask = ~np.isnan(obj_points).any(axis=1)
        valid_indices_before_unique = np.where(valid_mask)[0]

        if not np.any(valid_mask):
            return np.array([]), np.array([]), []

        img_points_intermediate = centroids[valid_mask]
        obj_points_intermediate = obj_points[valid_mask]

        unique_coords, unique_indices_in_intermediate = np.unique(
            obj_points_intermediate, axis=0, return_index=True
        )

        img_points = img_points_intermediate[unique_indices_in_intermediate]
        valid_obj_points = unique_coords

        final_original_indices = valid_indices_before_unique[
            unique_indices_in_intermediate
        ]

        correspondences = []
        ref_point_final_idx = -1

        if origin_idx in final_original_indices:
            found_indices = np.where(final_original_indices == origin_idx)[0]
            if len(found_indices) > 0:
                ref_point_final_idx = found_indices[0]

        for i in range(len(img_points)):
            is_ref = i == ref_point_final_idx
            correspondence = {
                "id": i,
                "img_point": img_points[i].astype(np.float32),
                "obj_point": valid_obj_points[i].astype(np.float32),
                "is_reference": is_ref,
            }
            correspondences.append(correspondence)

        return (
            img_points.astype(np.float32),
            valid_obj_points.astype(np.float32),
            correspondences,
        )

    @staticmethod
    def get_dists(points, target_point, k):
        """Get k nearest neighbors and distances for a target point.

        Args:
            points: Array of points to search
            target_point: Target point for nearest neighbors
            k: Number of neighbors to return
        Returns:
            distances: Array of distances to nearest neighbors
            indices: Array of indices of nearest neighbors
        """
        from sklearn.neighbors import NearestNeighbors

        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if len(points) == 0:
            return np.array([]), np.array([], dtype=int)

        target_point = np.asarray(target_point)
        if target_point.ndim == 1:
            target_point = target_point.reshape(1, -1)

        actual_k = min(k, len(points))
        if actual_k == 0:
            return np.array([]), np.array([], dtype=int)

        nbrs = NearestNeighbors(n_neighbors=actual_k).fit(points)
        distances, indices = nbrs.kneighbors(target_point)
        return distances[0], indices[0]
