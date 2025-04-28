import os
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from pycalib.feature_processing.feature_processing_utils import rvec_to_euler
from pycalib.optimization.jacobian_builder_stereo import JacobianBuilderStereo
from pycalib.optimization.optimizer_configs import (
    Extrinsics,
    Intrinsics,
    OptimizationState,
    OptimizerParams,
    ParamsGuess,
    StereoCalibrationFlags,
    setup_logger,
)

logger = setup_logger(logger_name="stereo_calibration")


class StereoOptimizer:
    """
    Stereo camera calibration optimizer that jointly optimizes parameters
    of two cameras and their relative pose.
    """

    def __init__(
        self,
        left_resolution: Tuple[int, int],
        right_resolution: Tuple[int, int],
        left_feature_data: Dict,
        right_feature_data: Dict,
        left_calib_data: Dict,
        right_calib_data: Dict,
        optimizer_params: OptimizerParams = None,
        flags: StereoCalibrationFlags = None,
    ):
        self.optimizer_params = optimizer_params or OptimizerParams()
        self.flags = flags or StereoCalibrationFlags()

        self.left_flags = self.flags.left_flags
        self.right_flags = self.flags.right_flags

        self.left_resolution = left_resolution
        self.right_resolution = right_resolution

        self.optim_state = OptimizationState()

        left_params_guess = self._create_params_guess(left_resolution, left_calib_data)
        right_params_guess = self._create_params_guess(
            right_resolution, right_calib_data
        )

        self.left_intrinsics = Intrinsics(
            fx=left_params_guess.fx,
            fy=left_params_guess.fy,
            cx=left_params_guess.cx,
            cy=left_params_guess.cy,
            dist_coeffs=left_params_guess.dist_coeffs,
        )

        self.right_intrinsics = Intrinsics(
            fx=right_params_guess.fx,
            fy=right_params_guess.fy,
            cx=right_params_guess.cx,
            cy=right_params_guess.cy,
            dist_coeffs=right_params_guess.dist_coeffs,
        )

        self.rel_rvec = np.zeros((3, 1))
        self.rel_tvec = np.zeros((3, 1))
        self.initial_rel_rvec = np.zeros((3, 1))
        self.initial_rel_tvec = np.zeros((3, 1))

        self.left_extrinsics = left_calib_data["extrinsics"]
        self.right_extrinsics = right_calib_data["extrinsics"]

        self._load_and_align_feature_data(
            left_feature_data, right_feature_data, left_calib_data, right_calib_data
        )

        self.all_left_obj_points = None
        self.all_left_img_points = None
        self.all_right_obj_points = None
        self.all_right_img_points = None
        self._pack_all_points()

        if self.valid_frames:
            self._estimate_init_rel_pose()
        else:
            logger.warning(
                "[__init__] No valid frames found after alignment. Cannot estimate initial relative pose."
            )

        self.jacobian_builder = JacobianBuilderStereo(self)

        self._cost_history = []
        self._iter_count = 0
        base_plot_output_dir = "pycalib/results/debugging/stereo"
        self._plot_output_dir_left = os.path.join(base_plot_output_dir, "left")
        self._plot_output_dir_right = os.path.join(base_plot_output_dir, "right")
        os.makedirs(self._plot_output_dir_left, exist_ok=True)
        os.makedirs(self._plot_output_dir_right, exist_ok=True)

    def _load_and_align_feature_data(
        self, left_feature_data, right_feature_data, left_calib_data, right_calib_data
    ):
        """Loads feature points, finds common valid frames, and aligns points."""
        self.left_image_points = left_feature_data.get("image_points", {})
        self.right_image_points = right_feature_data.get("image_points", {})
        self.left_object_points = left_feature_data.get("object_points", {})
        self.right_object_points = right_feature_data.get("object_points", {})

        valid_frames_left = left_calib_data.get("valid_frames", [])
        valid_frames_right = right_calib_data.get("valid_frames", [])

        left_available = set(self.left_object_points.keys())
        right_available = set(self.right_object_points.keys())

        left_initial_valid = (
            set(valid_frames_left) if valid_frames_left else left_available
        )
        right_initial_valid = (
            set(valid_frames_right) if valid_frames_right else right_available
        )

        common_initial_frames = sorted(list(left_initial_valid & right_initial_valid))

        if not common_initial_frames:
            logger.error(
                "[StereoOptimizer._load_and_align_feature_data] No common frames found between left and right datasets based on initial data."
            )
            self.valid_frames = []
            self.discarded_frames = []
            return

        self.valid_frames, self.discarded_frames = self._align_feature_points(
            common_initial_frames
        )

        logger.info(
            f"[StereoOptimizer._load_and_align_feature_data] Found {len(self.valid_frames)} valid frames after alignment."
        )

    def _align_feature_points(self, frames_to_check: List[int]):
        """
        Aligns 3D object points between left and right views for given frames.
        Modifies self.left/right_object/image_points dictionaries in place.

        Args:
            frames_to_check: List of frame indices to process.

        Returns:
            Tuple[List[int], List[int]]: Lists of successfully aligned frames and discarded frames.
        """
        aligned_frames = []
        discarded_frames = []

        for frame in frames_to_check:
            left_obj = self.left_object_points.get(frame)
            right_obj = self.right_object_points.get(frame)
            left_img = self.left_image_points.get(frame)
            right_img = self.right_image_points.get(frame)

            if any(
                p is None or len(p) == 0
                for p in [left_obj, right_obj, left_img, right_img]
            ):
                logger.warning(
                    f"[_align_feature_points] Frame {frame}: Skipping, missing or empty points data."
                )
                discarded_frames.append(frame)
                continue

            left_obj = np.asarray(left_obj, dtype=np.float64).reshape(-1, 3)
            right_obj = np.asarray(right_obj, dtype=np.float64).reshape(-1, 3)
            left_img = np.asarray(left_img, dtype=np.float64).reshape(-1, 2)
            right_img = np.asarray(right_img, dtype=np.float64).reshape(-1, 2)

            obj_scale = max(
                1.0, np.max([np.abs(left_obj).max(), np.abs(right_obj).max()])
            )
            tolerance = max(1e-4, obj_scale * 1e-5)

            left_tree = cKDTree(left_obj)
            right_tree = cKDTree(right_obj)
            l2r_dist, l2r_idx = right_tree.query(
                left_obj, distance_upper_bound=tolerance
            )
            r2l_dist, r2l_idx = left_tree.query(
                right_obj, distance_upper_bound=tolerance
            )

            valid_left_indices = []
            valid_right_indices = []
            for i, (dist, right_match_idx) in enumerate(zip(l2r_dist, l2r_idx)):
                if np.isfinite(dist):
                    if r2l_idx[right_match_idx] == i:
                        valid_left_indices.append(i)
                        valid_right_indices.append(right_match_idx)

            if len(valid_left_indices) < 6:  # Need at least 6 points for stability
                logger.warning(
                    f"[_align_feature_points] Frame {frame}: Skipping, not enough mutual matches ({len(valid_left_indices)} < 6)."
                )
                discarded_frames.append(frame)
                continue

            common_obj_points_from_left = left_obj[valid_left_indices]
            common_obj_points_from_right = right_obj[valid_right_indices]
            left_img_filtered = left_img[valid_left_indices]
            right_img_filtered = right_img[valid_right_indices]

            obj_diffs = np.linalg.norm(
                common_obj_points_from_left - common_obj_points_from_right, axis=1
            )
            if np.any(obj_diffs > tolerance):
                logger.warning(
                    f"[_align_feature_points] Frame {frame}: Skipping, high discrepancy in matched 3D points (max diff={np.max(obj_diffs):.1e} > tol={tolerance:.1e})."
                )
                discarded_frames.append(frame)
                continue

            common_obj_points = (
                common_obj_points_from_left + common_obj_points_from_right
            ) / 2.0

            self.left_object_points[frame] = common_obj_points
            self.left_image_points[frame] = left_img_filtered
            self.right_object_points[frame] = common_obj_points
            self.right_image_points[frame] = right_img_filtered

            aligned_frames.append(frame)
            logger.info(
                f"[_align_feature_points] Frame {frame}: Aligned {len(common_obj_points)} points."
            )

        for frame in discarded_frames:
            self.left_object_points.pop(frame, None)
            self.left_image_points.pop(frame, None)
            self.right_object_points.pop(frame, None)
            self.right_image_points.pop(frame, None)
            self.left_extrinsics.pop(frame, None)
            self.right_extrinsics.pop(frame, None)

        return sorted(aligned_frames), sorted(list(set(discarded_frames)))

    def _pack_all_points(self):
        self.all_left_obj_points = []
        self.all_left_img_points = []
        self.all_right_obj_points = []
        self.all_right_img_points = []

        for frame in self.valid_frames:
            if frame in self.left_object_points and frame in self.left_image_points:
                self.all_left_obj_points.append(self.left_object_points[frame])
                self.all_left_img_points.append(self.left_image_points[frame])
            if frame in self.right_object_points and frame in self.right_image_points:
                self.all_right_obj_points.append(self.right_object_points[frame])
                self.all_right_img_points.append(self.right_image_points[frame])

        if (
            not self.all_left_obj_points
            or not self.all_left_img_points
            or not self.all_right_obj_points
            or not self.all_right_img_points
        ):
            raise ValueError("[_pack_all_points] No valid points found in any frame.")

        self.all_left_obj_points = np.vstack(self.all_left_obj_points)
        self.all_left_img_points = np.vstack(self.all_left_img_points)
        self.all_right_obj_points = np.vstack(self.all_right_obj_points)
        self.all_right_img_points = np.vstack(self.all_right_img_points)

    def _estimate_init_rel_pose(self):
        """
        Estimate initial relative pose between left and right cameras
        """
        if not self.valid_frames:
            raise ValueError(
                "[_estimate_init_relative_pose] No valid frames to estimate relative pose from."
            )

        left_extr = self.left_extrinsics
        right_extr = self.right_extrinsics

        if (
            left_extr.rvec is None
            or left_extr.tvec is None
            or right_extr.rvec is None
            or right_extr.tvec is None
        ):
            logger.warning("[_estimate_init_rel_pose] Incomplete initial extrinsics.")
            raise ValueError("[_estimate_init_rel_pose] Incomplete initial extrinsics.")

        R_left = cv2.Rodrigues(left_extr.rvec)[0]
        R_right = cv2.Rodrigues(right_extr.rvec)[0]
        t_left = left_extr.tvec
        t_right = right_extr.tvec

        R_rel = R_right @ R_left.T
        tvec_rel = t_right - R_rel @ t_left

        rvec_rel, _ = cv2.Rodrigues(R_rel)

        self.rel_rvec = rvec_rel.reshape(3, 1)
        self.rel_tvec = tvec_rel.reshape(3, 1)

        logger.info(
            f"[_estimate_init_rel_pose] Initial rel rvec: {self.rel_rvec.flatten()}, euler deg: {rvec_to_euler(self.rel_rvec)}"
        )
        logger.info(
            f"[_estimate_init_rel_pose] Initial rel tvec: {self.rel_tvec.flatten()}"
        )
        print(f"Initial rel rotation (deg): {rvec_to_euler(self.rel_rvec)}")
        print(f"Initial rel translation: {self.rel_tvec.flatten()}")

        self.initial_rel_rvec = self.rel_rvec.copy()
        self.initial_rel_tvec = self.rel_tvec.copy()

    def _left_to_right_extrinsics(self, left_rvec, left_tvec, rel_rvec, rel_tvec):
        """
        Calculate right camera extrinsics (world-to-right) given left camera
        extrinsics (world-to-left) and the relative pose (left-to-right).
        """
        left_rvec = np.asarray(left_rvec, dtype=np.float64).reshape(3, 1)
        left_tvec = np.asarray(left_tvec, dtype=np.float64).reshape(3, 1)
        rel_rvec = np.asarray(rel_rvec, dtype=np.float64).reshape(3, 1)
        rel_tvec = np.asarray(rel_tvec, dtype=np.float64).reshape(3, 1)

        if np.any(np.isnan([left_rvec, left_tvec, rel_rvec, rel_tvec])) or np.any(
            np.isinf([left_rvec, left_tvec, rel_rvec, rel_tvec])
        ):
            raise ValueError("Invalid input vectors containing NaN or Inf values")

        R_left, _ = cv2.Rodrigues(left_rvec)
        R_rel, _ = cv2.Rodrigues(rel_rvec)

        R_right = R_rel @ R_left

        tvec_right = R_rel @ left_tvec + rel_tvec

        trans_magnitude = np.linalg.norm(tvec_right)
        MAX_TRANS_MAG = 1000.0
        if trans_magnitude > MAX_TRANS_MAG:
            raise ValueError(
                f"[_left_to_right_extrinsics] Calculated right tvec magnitude is large: {trans_magnitude:.2f}"
            )

        rvec_right, _ = cv2.Rodrigues(R_right)

        return Extrinsics(rvec=rvec_right.reshape(3, 1), tvec=tvec_right.reshape(3, 1))

    def _pack_parameters(self):
        self._param_indices = {}
        params_list = []
        current_idx = 0

        # 1. Left Intrinsics (if optimizing)
        num_left_intr_params = 0
        start_left_intr = len(params_list)
        if self.left_flags:
            if self.left_flags.estimate_focal:
                params_list.extend([self.left_intrinsics.fx, self.left_intrinsics.fy])
            if self.left_flags.estimate_principal:
                params_list.extend([self.left_intrinsics.cx, self.left_intrinsics.cy])
            left_dist_flags = [
                self.left_flags.estimate_k1,
                self.left_flags.estimate_k2,
                self.left_flags.estimate_p1,
                self.left_flags.estimate_p2,
                self.left_flags.estimate_k3,
            ]
            if np.any(left_dist_flags):
                params_list.extend(self.left_intrinsics.dist_coeffs[left_dist_flags])
            num_left_intr_params = len(params_list) - start_left_intr
        self._param_indices["left_intrinsics"] = (
            current_idx,
            current_idx + num_left_intr_params,
        )
        current_idx += num_left_intr_params

        num_right_intr_params = 0
        start_right_intr = len(params_list)
        if self.right_flags:
            if self.right_flags.estimate_focal:
                params_list.extend([self.right_intrinsics.fx, self.right_intrinsics.fy])
            if self.right_flags.estimate_principal:
                params_list.extend([self.right_intrinsics.cx, self.right_intrinsics.cy])
            right_dist_flags = [
                self.right_flags.estimate_k1,
                self.right_flags.estimate_k2,
                self.right_flags.estimate_p1,
                self.right_flags.estimate_p2,
                self.right_flags.estimate_k3,
            ]
            if np.any(right_dist_flags):
                params_list.extend(self.right_intrinsics.dist_coeffs[right_dist_flags])
            num_right_intr_params = len(params_list) - start_right_intr
        self._param_indices["right_intrinsics"] = (
            current_idx,
            current_idx + num_right_intr_params,
        )
        current_idx += num_right_intr_params

        params_list.extend(self.rel_rvec.flatten())
        params_list.extend(self.rel_tvec.flatten())
        self._param_indices["relative_pose"] = (current_idx, current_idx + 6)
        current_idx += 6

        self._param_indices["left_extrinsics"] = {}
        params_list.extend(self.left_extrinsics.rvec.flatten())
        params_list.extend(self.left_extrinsics.tvec.flatten())
        self._param_indices["left_extrinsics"] = (current_idx, current_idx + 6)
        current_idx += 6

        return np.array(params_list, dtype=np.float64)

    def _unpack_parameters(self, params: np.ndarray):
        if not hasattr(self, "_param_indices") or not self._param_indices:
            raise ValueError(
                "[_unpack_parameters] Parameter indices missing. Call _pack_parameters first."
            )

        start_idx, end_idx = self._param_indices["left_intrinsics"]
        if start_idx != end_idx:
            intr_params = params[start_idx:end_idx]
            current_intr_idx = 0
            if self.left_flags:
                if self.left_flags.estimate_focal:
                    self.left_intrinsics.fx = intr_params[current_intr_idx]
                    self.left_intrinsics.fy = intr_params[current_intr_idx + 1]
                    current_intr_idx += 2
                if self.left_flags.estimate_principal:
                    self.left_intrinsics.cx = intr_params[current_intr_idx]
                    self.left_intrinsics.cy = intr_params[current_intr_idx + 1]
                    current_intr_idx += 2
                left_dist_flags = [
                    self.left_flags.estimate_k1,
                    self.left_flags.estimate_k2,
                    self.left_flags.estimate_p1,
                    self.left_flags.estimate_p2,
                    self.left_flags.estimate_k3,
                ]
                num_dist = np.sum(left_dist_flags)
                if num_dist > 0:
                    self.left_intrinsics.dist_coeffs[left_dist_flags] = intr_params[
                        current_intr_idx : current_intr_idx + num_dist
                    ]
                    current_intr_idx += num_dist

        start_idx, end_idx = self._param_indices["right_intrinsics"]
        if start_idx != end_idx:
            intr_params = params[start_idx:end_idx]
            current_intr_idx = 0
            if self.right_flags:
                if self.right_flags.estimate_focal:
                    self.right_intrinsics.fx = intr_params[current_intr_idx]
                    self.right_intrinsics.fy = intr_params[current_intr_idx + 1]
                    current_intr_idx += 2
                if self.right_flags.estimate_principal:
                    self.right_intrinsics.cx = intr_params[current_intr_idx]
                    self.right_intrinsics.cy = intr_params[current_intr_idx + 1]
                    current_intr_idx += 2
                right_dist_flags = [
                    self.right_flags.estimate_k1,
                    self.right_flags.estimate_k2,
                    self.right_flags.estimate_p1,
                    self.right_flags.estimate_p2,
                    self.right_flags.estimate_k3,
                ]
                num_dist = np.sum(right_dist_flags)
                if num_dist > 0:
                    self.right_intrinsics.dist_coeffs[right_dist_flags] = intr_params[
                        current_intr_idx : current_intr_idx + num_dist
                    ]
                    current_intr_idx += num_dist

        start_idx, end_idx = self._param_indices["relative_pose"]
        rel_pose_params = params[start_idx:end_idx]
        self.rel_rvec = rel_pose_params[:3].reshape(3, 1)
        self.rel_tvec = rel_pose_params[3:].reshape(3, 1)

        start_idx, end_idx = self._param_indices["left_extrinsics"]
        ext_params = params[start_idx:end_idx]
        self.left_extrinsics.rvec = ext_params[:3].reshape(3, 1)
        self.left_extrinsics.tvec = ext_params[3:].reshape(3, 1)

    def _calc_error_left(self):
        """Compute reprojection error for the left camera for a single frame."""
        if (
            self.all_left_obj_points is None
            or self.all_left_img_points is None
            or self.left_extrinsics is None
        ):
            logger.warning(
                "[_calc_single_frame_error_left] Missing data (points or extrinsics)."
            )
            raise ValueError(
                "[_calc_single_frame_error_left] Missing data (points or extrinsics)."
            )

        rvec = np.asarray(self.left_extrinsics.rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(self.left_extrinsics.tvec, dtype=np.float64).reshape(3, 1)

        camera_matrix = np.array(
            [
                [self.left_intrinsics.fx, 0, self.left_intrinsics.cx],
                [0, self.left_intrinsics.fy, self.left_intrinsics.cy],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        projected_points, _ = cv2.projectPoints(
            self.all_left_obj_points,
            rvec,
            tvec,
            camera_matrix,
            self.left_intrinsics.dist_coeffs,
        )

        errors = projected_points.reshape(-1, 2) - self.all_left_img_points.reshape(
            -1, 2
        )

        error_norms = np.linalg.norm(errors, axis=1)
        MAX_ERR_WARN = 200.0
        if np.any(error_norms > MAX_ERR_WARN):
            max_err = np.max(error_norms)
            logger.warning(
                f"[_calc_single_frame_error_left] Large error detected (max={max_err:.2f} > {MAX_ERR_WARN}px)."
            )

        return errors.flatten()

    def _calc_error_right(self):
        """Compute reprojection error for the right camera for a single frame."""
        if (
            self.all_right_obj_points is None
            or self.all_right_img_points is None
            or self.left_extrinsics is None
        ):
            logger.warning(
                "[_calc_single_frame_error_right] Missing data (points or left extrinsics)."
            )
            return np.array([])

        extr_right = self._left_to_right_extrinsics(
            self.left_extrinsics.rvec,
            self.left_extrinsics.tvec,
            self.rel_rvec,
            self.rel_tvec,
        )
        rvec_right = extr_right.rvec
        tvec_right = extr_right.tvec
        if (
            np.any(np.isnan(rvec_right))
            or np.any(np.isnan(tvec_right))
            or np.any(np.isinf(rvec_right))
            or np.any(np.isinf(tvec_right))
        ):
            logger.warning(
                "[_calc_single_frame_error_right] Calculated right extrinsics are invalid (NaN/Inf). Using zeros."
            )
            raise ValueError(
                "[_calc_single_frame_error_right] Calculated right extrinsics are invalid (NaN/Inf)."
            )

        camera_matrix = np.array(
            [
                [self.right_intrinsics.fx, 0, self.right_intrinsics.cx],
                [0, self.right_intrinsics.fy, self.right_intrinsics.cy],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        projected_points, _ = cv2.projectPoints(
            self.all_right_obj_points,
            rvec_right,
            tvec_right,
            camera_matrix,
            self.right_intrinsics.dist_coeffs,
        )

        errors = projected_points.reshape(-1, 2) - self.all_right_img_points.reshape(
            -1, 2
        )

        return errors.flatten()

    def _calc_residuals(self, params: np.ndarray | None = None):
        """
        Calculate reprojection errors for both cameras.
        """
        if params is not None:
            self._unpack_parameters(params)

        residuals = np.concatenate([self._calc_error_left(), self._calc_error_right()])

        return residuals

    def _calibrate(self):
        self._store_initial_state()

        self._filter_outliers()

        initial_params = self._pack_parameters()

        logger.info(f"[_calibrate] Packed {initial_params.size} parameters.")

        self._validate_params_consistency(initial_params)

        self._cost_history = []
        self._iter_count = 0
        initial_errors = self._calc_residuals(initial_params)

        if initial_errors.size == 0:
            logger.error(
                "[_calibrate] Initial error calculation resulted in empty vector. Aborting."
            )
            raise ValueError(
                "[_calibrate] Initial error calculation resulted in empty vector."
            )

        initial_cost = 0.5 * np.sum(initial_errors**2)
        self._cost_history.append(initial_cost)

        if not np.isfinite(initial_cost):
            logger.error(
                f"[_calibrate] Failed to calculate a valid initial cost ({initial_cost}). Initial errors sum of squares: {np.sum(initial_errors**2)}"
            )
            logger.debug(
                f"[_calibrate] Sample initial errors (first 10): {initial_errors[:10]}"
            )
            raise ValueError("[_calibrate] Failed to calculate a valid initial cost.")

        self.optim_state.initial_cost = initial_cost
        logger.info(f"[_calibrate] Initial Cost: {initial_cost:.6e}")

        logger.info(
            f"[_calibrate] Starting optimization with {self.optimizer_params.max_iter} iterations and {self.optimizer_params.opt_method} method..."
        )

        result = least_squares(
            self._calc_residuals,
            initial_params,
            jac=self.jacobian_builder.compute_full_jacobian,
            method=self.optimizer_params.opt_method,
            ftol=self.optimizer_params.ftol,
            xtol=self.optimizer_params.xtol,
            gtol=self.optimizer_params.gtol,
            max_nfev=self.optimizer_params.max_iter,
            verbose=self.optimizer_params.verbose,
        )

        self._iter_count = result.njev if hasattr(result, "njev") else result.nfev

        logger.info(f"  Status: {result.status} ({result.message})")
        logger.info(
            f"  Iterations (Jacobian evals): {result.njev if hasattr(result, 'njev') else 'N/A'}"
        )
        logger.info(f"  Function evaluations: {result.nfev}")

        self._unpack_parameters(result.x)

        final_cost = (
            self._cost_history[-1]
            if self._cost_history and np.isfinite(self._cost_history[-1])
            else np.nan
        )

        logger.info(f"  Final Cost:   {final_cost:.6e}")

        self.optim_state.final_result = result
        self.optim_state.best_params = result.x
        self.optim_state.best_error = final_cost
        self.optim_state.current_error = final_cost

        self._plot_optimization_history()
        results = self._collect_results()
        self.save_results(results)

        return results

    def _filter_outliers(self):
        """
        Filter out outlier points that might affect optimization convergence.
        """
        percentile_threshold = 70
        logger.info(
            f"[_filter_outliers] Filtering outliers above {percentile_threshold}th percentile"
        )

        if self.all_left_obj_points is None or self.all_left_obj_points.shape[0] == 0:
            logger.warning("[_filter_outliers] No points available to filter.")
            return 0

        left_camera_matrix = np.array(
            [
                [self.left_intrinsics.fx, 0, self.left_intrinsics.cx],
                [0, self.left_intrinsics.fy, self.left_intrinsics.cy],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        left_projected_points, _ = cv2.projectPoints(
            self.all_left_obj_points,
            self.left_extrinsics.rvec,
            self.left_extrinsics.tvec,
            left_camera_matrix,
            self.left_intrinsics.dist_coeffs,
        )
        left_errors = np.linalg.norm(
            left_projected_points.reshape(-1, 2) - self.all_left_img_points, axis=1
        )

        try:
            right_extr = self._left_to_right_extrinsics(
                self.left_extrinsics.rvec,
                self.left_extrinsics.tvec,
                self.rel_rvec,
                self.rel_tvec,
            )
            right_camera_matrix = np.array(
                [
                    [self.right_intrinsics.fx, 0, self.right_intrinsics.cx],
                    [0, self.right_intrinsics.fy, self.right_intrinsics.cy],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )
            right_projected_points, _ = cv2.projectPoints(
                self.all_right_obj_points,
                right_extr.rvec,
                right_extr.tvec,
                right_camera_matrix,
                self.right_intrinsics.dist_coeffs,
            )
            right_errors = np.linalg.norm(
                right_projected_points.reshape(-1, 2) - self.all_right_img_points,
                axis=1,
            )
        except ValueError as e:
            logger.error(
                f"[_filter_outliers] Error calculating initial right projections: {e}. Cannot filter based on right camera errors."
            )
            raise

        combined_errors = np.maximum(left_errors, right_errors)

        initial_point_count = self.all_left_obj_points.shape[0]
        points_to_remove = initial_point_count - np.sum(
            combined_errors <= np.percentile(combined_errors, percentile_threshold)
        )

        if points_to_remove > 0:
            logger.info(
                f"[_filter_outliers] Removed {points_to_remove} outlier points with combined errors > {np.percentile(combined_errors, percentile_threshold):.3f} px. "
                f"Remaining points: {self.all_left_obj_points.shape[0]}"
            )
        else:
            logger.info(
                f"[_filter_outliers] No points removed. Threshold ({percentile_threshold}th percentile) was {np.percentile(combined_errors, percentile_threshold):.3f} px."
            )

        if not (
            self.all_left_obj_points.shape[0]
            == self.all_left_img_points.shape[0]
            == self.all_right_obj_points.shape[0]
            == self.all_right_img_points.shape[0]
        ):
            logger.error("Point array shapes inconsistent after filtering!")
            raise RuntimeError("Inconsistent point data after outlier filtering.")

        return points_to_remove

    def _validate_params_consistency(self, initial_params):
        self._unpack_parameters(initial_params)
        repacked_params = self._pack_parameters()

        if not np.allclose(initial_params, repacked_params, atol=1e-8, rtol=1e-8):
            diff_indices = np.where(
                ~np.isclose(initial_params, repacked_params, atol=1e-8, rtol=1e-8)
            )[0]
            max_diff = np.max(np.abs(initial_params - repacked_params))
            error_msg = (
                f"[_validate_params_consistency] Pack/Unpack consistency check failed! Max difference: {max_diff:.2e}. "
                f"Indices with differences (first 10): {diff_indices[:10]}"
            )
            logger.error(error_msg)
            logger.error(
                f"[_validate_params_consistency] Original params sample (at first diff): {initial_params[diff_indices[0]]}"
            )
            logger.error(
                f"[_validate_params_consistency] Repacked params sample (at first diff): {repacked_params[diff_indices[0]]}"
            )

    def _store_initial_state(self):
        self._initial_state = {
            "left_intrinsics": self.left_intrinsics,
            "right_intrinsics": self.right_intrinsics,
            "left_extrinsics": self.left_extrinsics,
            "rel_rvec": self.rel_rvec,
            "rel_tvec": self.rel_tvec,
        }

    def _collect_results(self):
        results = {
            "rms_error_px": np.nan,
            "relative_pose": {},
            "left_camera": {"intrinsics": {}, "extrinsics": {}},
            "right_camera": {"intrinsics": {}, "extrinsics": {}},
            "metadata": {},
        }

        results["left_camera"]["intrinsics"] = self.left_intrinsics.to_dict()
        results["right_camera"]["intrinsics"] = self.right_intrinsics.to_dict()

        results["relative_pose"]["rvec"] = self.rel_rvec.tolist()
        results["relative_pose"]["tvec"] = self.rel_tvec.tolist()

        R_rel, _ = cv2.Rodrigues(self.rel_rvec)
        results["relative_pose"]["rotation_matrix"] = R_rel.tolist()
        results["relative_pose"]["euler_angles_deg"] = rvec_to_euler(self.rel_rvec)
        results["left_camera"]["extrinsics"] = self.left_extrinsics.to_dict()

        right_extrinsics = self._left_to_right_extrinsics(
            self.left_extrinsics.rvec,
            self.left_extrinsics.tvec,
            self.rel_rvec,
            self.rel_tvec,
        )
        results["right_camera"]["extrinsics"] = right_extrinsics.to_dict()

        results["metadata"]["left_resolution"] = self.left_resolution
        results["metadata"]["right_resolution"] = self.right_resolution

        num_points = (
            self.all_left_obj_points.shape[0]
            if self.all_left_obj_points is not None
            else 0
        )
        results["metadata"]["total_points_per_camera"] = num_points
        results["metadata"]["total_observations"] = num_points * 2

        final_params = self._pack_parameters()
        errors = self._calc_residuals(final_params)
        rms_error = np.sqrt(np.mean(errors**2)) if errors.size > 0 else 0.0
        results["rms_error_px"] = float(rms_error)

        return results

    def _create_params_guess(self, resolution, calib_data):
        intr_data = calib_data.get("intrinsics", {})

        if isinstance(intr_data, Intrinsics):
            params_guess = ParamsGuess(
                image_size=resolution,
                fx=intr_data.fx,
                fy=intr_data.fy,
                cx=intr_data.cx,
                cy=intr_data.cy,
                dist_coeffs=intr_data.dist_coeffs,
            )
        elif isinstance(intr_data, dict):
            params_guess = ParamsGuess(
                image_size=resolution,
                fx=intr_data.get("fx"),
                fy=intr_data.get("fy"),
                cx=intr_data.get("cx"),
                cy=intr_data.get("cy"),
                dist_coeffs=np.array(
                    intr_data.get("dist_coeffs", [0.0] * 5), dtype=np.float64
                ).flatten()[:5],
            )
            if len(params_guess.dist_coeffs) < 5:
                params_guess.dist_coeffs = np.pad(
                    params_guess.dist_coeffs, (0, 5 - len(params_guess.dist_coeffs))
                )

        else:
            logger.warning(
                f"[_create_params_guess] Unexpected intrinsics data type: {type(intr_data)}. Using default ParamsGuess."
            )
            params_guess = ParamsGuess(image_size=resolution)

        logger.info(
            f"[_create_params_guess] Initial guess for {resolution}: fx={params_guess.fx:.2f}, fy={params_guess.fy:.2f}, cx={params_guess.cx:.2f}, cy={params_guess.cy:.2f}"
        )
        return params_guess

    def _plot_optimization_history(self):
        valid_costs = np.array(
            [cost for cost in self._cost_history if np.isfinite(cost)]
        )

        if valid_costs.size == 0:
            logger.warning(
                "[Plotting] No valid (finite) cost values found in history, cannot plot."
            )
            return

        evaluations = np.arange(len(self._cost_history))
        valid_eval_indices = np.where(np.isfinite(self._cost_history))[0]

        save_path = os.path.join(
            os.path.dirname(self._plot_output_dir_left), "stereo_cost_history.png"
        )

        plt.figure(figsize=(10, 6))
        plt.plot(
            evaluations[valid_eval_indices], valid_costs, marker=".", linestyle="-"
        )
        plt.xlabel("Function Evaluation")
        plt.ylabel("Cost (0.5 * Sum of Squared Errors)")
        plt.title("Stereo Optimization Cost History")
        plt.grid(True)

        if (
            valid_costs.size > 1
            and (valid_costs.max() / max(valid_costs.min(), 1e-12)) > 10
        ):
            min_positive_cost = (
                np.min(valid_costs[valid_costs > 0])
                if np.any(valid_costs > 0)
                else 1e-9
            )
            plt.ylim(bottom=max(min_positive_cost * 0.1, 1e-9))
            plt.yscale("log")

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"[Plotting] Cost history plot saved to {save_path}")

    def save_results(self, results):
        import json

        os.makedirs("pycalib/results", exist_ok=True)
        with open("pycalib/results/stereo_calib_report.json", "w") as f:
            json.dump(results, f, indent=2)
