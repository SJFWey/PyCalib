import os
from collections import defaultdict

import cv2
import numpy as np
from scipy.optimize import least_squares

from pycalib.optimization.jacobian_builder_single import JacobianBuilderSingle
from pycalib.optimization.optimizer_configs import (
    Extrinsics,
    Intrinsics,
    OptimizationState,
    OptimizerFlags,
    OptimizerParams,
    ParamsGuess,
    setup_logger,
)
from pycalib.optimization.optimizer_utils import (
    plot_cost_history,
    visualize_reproj_per_frame,
)


class Optimizer:
    """
    Optimizer class for minimizing the reprojection error.
    """

    def __init__(
        self,
        cam_name: str,
        calib_data: dict,
        params_guess: ParamsGuess | None = None,
        optimizer_params: OptimizerParams | None = None,
        flags: OptimizerFlags | None = None,
    ):
        self.cam_name = cam_name
        self.calib_data = calib_data
        self.optimizer_params = optimizer_params or OptimizerParams()
        self.flags = flags or OptimizerFlags()
        self.params_guess = params_guess or ParamsGuess()

        global logger
        logger = setup_logger(cam_name=cam_name)

        logger.info(f"[__init__] Starting calibration for camera: {cam_name}")
        logger.info(f"[__init__] Image size: {self.params_guess.image_size}")

        flags_dict = flags.to_dict() if flags else {}
        self.optim_flags = {
            "focal": flags_dict.get("focal", False),
            "principal_point": flags_dict.get("principal_point", False),
            "distortion": flags_dict.get("distortion", False),
            "extrinsics": flags_dict.get("extrinsics", False),
        }

        self.image_size = self.params_guess.image_size
        self.fx = self.params_guess.fx
        self.fy = self.params_guess.fy
        self.cx = self.params_guess.cx
        self.cy = self.params_guess.cy
        self.dist_coeffs = self.params_guess.dist_coeffs

        self.valid_frames = set()
        self.discarded_frames = set()

        self._load_feature_data(calib_data)
        self.all_obj_points = None
        self.all_img_points = None
        self.num_points_per_frame = []
        self._pack_all_points()

        self.optim_state = OptimizationState()

        self.intrinsics = Intrinsics(
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            dist_coeffs=self.dist_coeffs,
        )

        self.extrinsics: Extrinsics = None

        self._estimate_init_params()

        self.optim_state.params_history.append(self._pack_parameters())

        self.jacobian_builder = JacobianBuilderSingle()

    def _pack_all_points(self):
        obj_points_list = []
        img_points_list = []

        for frame in sorted(list(self.valid_frames)):
            if frame in self.obj_pts_all_frames and frame in self.img_pts_all_frames:
                self.num_points_per_frame.append(len(self.obj_pts_all_frames[frame]))
                obj_points = self.obj_pts_all_frames[frame]
                img_points = self.img_pts_all_frames[frame]

                if obj_points.shape[0] > 0 and img_points.shape[0] > 0:
                    obj_points_list.append(obj_points)
                    img_points_list.append(img_points)

        if not obj_points_list:
            raise ValueError("No valid points found in any frame.")

        self.all_obj_points = np.vstack(obj_points_list)
        self.all_img_points = np.vstack(img_points_list)

    def _estimate_init_params(self):
        """
        Perform initial parameter estimation using all points.
        """
        if self.all_obj_points is None or self.all_img_points is None:
            self._pack_all_points()

        logger.info(
            f"[_estimate_init_params] Calibrated intrinsics: fx={self.intrinsics.fx:.2f}, fy={self.intrinsics.fy:.2f}, cx={self.intrinsics.cx:.2f}, cy={self.intrinsics.cy:.2f}"
        )
        print(
            f"Initial estimated: fx={self.intrinsics.fx:.2f}, fy={self.intrinsics.fy:.2f}, cx={self.intrinsics.cx:.2f}, cy={self.intrinsics.cy:.2f}"
        )

        camera_matrix = np.array(
            [
                [self.intrinsics.fx, 0, self.intrinsics.cx],
                [0, self.intrinsics.fy, self.intrinsics.cy],
                [0, 0, 1],
            ],
        )

        # error, camera_matrix, dist_coeffs, rvec, tvec = cv2.calibrateCamera(
        #     [self.all_obj_points],
        #     [self.all_img_points],
        #     self.image_size,
        #     camera_matrix,
        #     self.intrinsics.dist_coeffs,
        #     flags=cv2.CALIB_USE_INTRINSIC_GUESS,
        # )

        success, rvec, tvec = cv2.solvePnP(
            self.all_obj_points,
            self.all_img_points,
            camera_matrix,
            self.intrinsics.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if success:
            rvec, tvec = cv2.solvePnPRefineLM(
                self.all_obj_points,
                self.all_img_points,
                camera_matrix,
                self.intrinsics.dist_coeffs,
                rvec,
                tvec,
            )
        self.extrinsics = Extrinsics(
            rvec=rvec.flatten(),
            tvec=tvec.flatten(),
        )

    def _pack_parameters(self):
        """Pack parameters into a single vector."""
        params = []

        if self.flags.estimate_extrinsics:
            params.extend(self.extrinsics.rvec.flatten())
            params.extend(self.extrinsics.tvec.flatten())

        if self.flags.estimate_focal:
            params.append(self.intrinsics.fx)
            params.append(self.intrinsics.fy)

        if self.flags.estimate_principal:
            params.append(self.intrinsics.cx)
            params.append(self.intrinsics.cy)

        if self.flags.estimate_k1:
            params.append(self.intrinsics.dist_coeffs[0])
        if self.flags.estimate_k2:
            params.append(self.intrinsics.dist_coeffs[1])
        if self.flags.estimate_p1:
            params.append(self.intrinsics.dist_coeffs[2])
        if self.flags.estimate_p2:
            params.append(self.intrinsics.dist_coeffs[3])
        if self.flags.estimate_k3:
            params.append(self.intrinsics.dist_coeffs[4])

        return np.array(params)

    def _unpack_parameters(self, params):
        """Unpack parameters from a single vector"""
        param_idx = 0

        if self.flags.estimate_extrinsics:
            for i in range(3):
                self.extrinsics.rvec[i] = params[param_idx]
                param_idx += 1
            for i in range(3):
                self.extrinsics.tvec[i] = params[param_idx]
                param_idx += 1

        if self.flags.estimate_focal:
            self.intrinsics.fx = params[param_idx]
            param_idx += 1
            self.intrinsics.fy = params[param_idx]
            param_idx += 1

        if self.flags.estimate_principal:
            self.intrinsics.cx = params[param_idx]
            param_idx += 1
            self.intrinsics.cy = params[param_idx]
            param_idx += 1

        if self.flags.estimate_k1:
            self.intrinsics.dist_coeffs[0] = params[param_idx]
            param_idx += 1
        if self.flags.estimate_k2:
            self.intrinsics.dist_coeffs[1] = params[param_idx]
            param_idx += 1
        if self.flags.estimate_p1:
            self.intrinsics.dist_coeffs[2] = params[param_idx]
            param_idx += 1
        if self.flags.estimate_p2:
            self.intrinsics.dist_coeffs[3] = params[param_idx]
            param_idx += 1
        if self.flags.estimate_k3:
            self.intrinsics.dist_coeffs[4] = params[param_idx]
            param_idx += 1

        return param_idx

    def _calc_residuals(self, params=None):
        if params is not None:
            self._unpack_parameters(params)

        if self.all_obj_points is None or self.all_img_points is None:
            self._pack_all_points()

        camera_matrix = np.array(
            [
                [self.intrinsics.fx, 0, self.intrinsics.cx],
                [0, self.intrinsics.fy, self.intrinsics.cy],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.asarray(self.intrinsics.dist_coeffs, dtype=np.float64).reshape(
            -1
        )

        rvec = np.asarray(self.extrinsics.rvec, dtype=np.float64).reshape(
            3,
        )
        tvec = np.asarray(self.extrinsics.tvec, dtype=np.float64).reshape(
            3,
        )

        projected_points, _ = cv2.projectPoints(
            self.all_obj_points, rvec, tvec, camera_matrix, dist_coeffs
        )
        projected_points = projected_points.reshape(-1, 2)
        errors_vec = projected_points - self.all_img_points

        return errors_vec.flatten()

    def _calibrate(self):
        if len(self.obj_pts_all_frames) == 0 or len(self.img_pts_all_frames) == 0:
            raise ValueError("No feature data available")

        self._filter_outliers()
        self._pack_all_points()

        params_pre_optim = self._pack_parameters()
        initial_errors = self._calc_residuals(params_pre_optim)
        initial_error_size = len(initial_errors)

        if initial_error_size == 0:
            raise ValueError(
                "[_calibrate] Initial error calculation resulted in zero residuals (perhaps all points were filtered?). Cannot optimize."
            )

        initial_cost = np.sum(initial_errors**2)
        self.optim_state.initial_cost = initial_cost
        logger.info(
            f"[_calibrate] Initial cost before optimization: {initial_cost:.4f}"
        )
        logger.info(f"[_calibrate] Expected residual vector size: {initial_error_size}")

        self.jacobian_builder.validate_params_consistency(
            self._pack_parameters, self._unpack_parameters
        )

        self.optim_state.cost_history = []

        def error_func_with_monitoring(p):
            self._unpack_parameters(p)
            residuals = self._calc_residuals(p)

            if residuals.size != initial_error_size:
                logger.error(
                    f"[error_func_with_monitoring] Residual size mismatch! Expected {initial_error_size}, Got {residuals.size}. This indicates a problem even after the fix."
                )
                raise ValueError(
                    f"[error_func_with_monitoring] Residual size mismatch. Expected {initial_error_size}, Got {residuals.size}."
                )

            cost = np.sum(residuals**2)
            self.optim_state.cost_history.append(cost)
            return residuals

        def jac_func(p):
            self._unpack_parameters(p)
            jacobian = self.jacobian_builder.compute_jacobian(
                self.all_obj_points,
                self.extrinsics,
                self.intrinsics,
                self.flags,
                self._unpack_parameters,
                p,
            )
            if jacobian.shape[0] != initial_error_size:
                logger.error(
                    f"Jacobian row count ({jacobian.shape[0]}) does not match expected residual size ({initial_error_size})."
                )
                raise ValueError("Jacobian size mismatch.")
            return jacobian

        params_for_optim = params_pre_optim

        logger.info(
            f"[_calibrate] Starting least_squares optimization with method='{self.optimizer_params.opt_method}'..."
        )

        result = least_squares(
            error_func_with_monitoring,
            params_for_optim,
            jac=jac_func,
            method=self.optimizer_params.opt_method,
            ftol=self.optimizer_params.ftol,
            xtol=self.optimizer_params.xtol,
            gtol=self.optimizer_params.gtol,
            max_nfev=self.optimizer_params.max_iter,
            verbose=self.optimizer_params.verbose,
        )
        logger.info(
            f"[_calibrate] Optimization finished. Status: {result.status}, Message: {result.message}"
        )
        self.optim_state.final_result = result
        self.optim_state.best_params = result.x
        self.optim_state.best_error = (
            self.optim_state.cost_history[-1] / (len(result.fun) / 2)
            if self.optim_state.cost_history and len(result.fun) > 0
            else np.sum(result.fun**2) / (len(result.fun) / 2)
            if len(result.fun) > 0
            else 0
        )
        logger.info(
            f"[_calibrate] Final Cost={self.optim_state.cost_history[-1]:.4f}"
            if self.optim_state.cost_history
            else f"[_calibrate] Final Cost={np.sum(result.fun**2):.4f}"
        )
        debug_dir = os.path.join("pycalib", "results", "debugging", "single")
        os.makedirs(debug_dir, exist_ok=True)
        plot_save_path = os.path.join(debug_dir, f"{self.cam_name}_cost_history.png")
        plot_cost_history(
            self.optim_state,
            self.cam_name,
            logger,
            save_path=plot_save_path,
            show_plot=False,
        )

        self.jacobian_builder.validate_params_consistency(
            self._pack_parameters, self._unpack_parameters
        )

        self.error_report = self._error_report()

        visualize_reproj_per_frame(
            cam_name=self.cam_name,
            valid_frames=self.valid_frames,
            obj_pts_all_frames=self.obj_pts_all_frames,
            img_pts_all_frames=self.img_pts_all_frames,
            intrinsics=self.intrinsics,
            extrinsics=self.extrinsics,
            image_size=self.image_size,
            logger=logger,
        )

        return {
            "intrinsics": self.intrinsics,
            "extrinsics": self.extrinsics,
            "errors": self.error_report,
            "optimization_result": self.optim_state.final_result,
        }

    def _error_report(self):
        report = {
            "mean_px": np.nan,
            "median_px": np.nan,
            "rms_px": np.nan,
            "max_px": np.nan,
            "min_px": np.nan,
            "per_frame_rms_px": {},
        }

        if (
            self.all_obj_points is not None
            and self.all_img_points is not None
            and len(self.all_obj_points) > 0
        ):
            camera_matrix = np.array(
                [
                    [self.intrinsics.fx, 0, self.intrinsics.cx],
                    [0, self.intrinsics.fy, self.intrinsics.cy],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )

            projected_points, _ = cv2.projectPoints(
                self.all_obj_points,
                self.extrinsics.rvec,
                self.extrinsics.tvec,
                camera_matrix,
                self.intrinsics.dist_coeffs,
            )
            projected_points = projected_points.reshape(-1, 2)

            errors_vec = projected_points - self.all_img_points
            errors_norm = np.linalg.norm(errors_vec, axis=1)

            report["mean_px"] = np.mean(errors_norm)
            report["median_px"] = np.median(errors_norm)
            report["rms_px"] = np.sqrt(np.mean(np.square(errors_norm)))
            report["max_px"] = np.max(errors_norm)
            report["min_px"] = np.min(errors_norm)

            total_points = len(errors_norm)
            logger.info(
                f"[error_report] Overall reprojection error stats ({total_points} points, px): "
                f"Mean={report['mean_px']:.3f}, RMS={report['rms_px']:.3f}, Median={report['median_px']:.3f}, "
                f"Min={report['min_px']:.3f}, Max={report['max_px']:.3f}"
            )

        for frame in self.valid_frames:
            if (
                frame not in self.obj_pts_all_frames
                or frame not in self.img_pts_all_frames
            ):
                logger.warning(f"[error_report] Skipping frame {frame}: Missing data.")
                report["per_frame_rms_px"][frame] = np.nan
                continue

            obj_points = self.obj_pts_all_frames[frame]
            img_points = self.img_pts_all_frames[frame]

            if obj_points.shape[0] == 0 or img_points.shape[0] == 0:
                logger.info(
                    f"[error_report] Frame {frame} has no points, skipping stats calculation."
                )
                report["per_frame_rms_px"][frame] = np.nan
                continue

            camera_matrix = np.array(
                [
                    [self.intrinsics.fx, 0, self.intrinsics.cx],
                    [0, self.intrinsics.fy, self.intrinsics.cy],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )

            projected_points, _ = cv2.projectPoints(
                obj_points,
                self.extrinsics.rvec,
                self.extrinsics.tvec,
                camera_matrix,
                self.intrinsics.dist_coeffs,
            )
            projected_points = projected_points.reshape(-1, 2)

            errors_vec = projected_points - img_points
            errors_norm = np.linalg.norm(errors_vec, axis=1)

            num_valid_points_frame = len(errors_norm)

            if num_valid_points_frame > 0:
                frame_rms_error = np.sqrt(np.mean(np.square(errors_norm)))
                report["per_frame_rms_px"][frame] = frame_rms_error
                logger.info(
                    f"[error_report] Frame {frame} reprojection error ({num_valid_points_frame} valid points): RMS={frame_rms_error:.3f} px"
                )
            else:
                logger.warning(
                    f"[error_report] Frame {frame}: No valid points after projection filtering. Skipping stats."
                )
        return report

    def _load_feature_data(self, calib_data):
        if "image_points" in calib_data.keys() and "object_points" in calib_data.keys():
            self.img_pts_all_frames = calib_data["image_points"]
            self.obj_pts_all_frames = calib_data["object_points"]
            for frame in set(self.img_pts_all_frames.keys()) | set(
                self.obj_pts_all_frames.keys()
            ):
                if len(self.img_pts_all_frames[frame]) == len(
                    self.obj_pts_all_frames[frame]
                ):
                    self.valid_frames.add(frame)
        else:
            self.img_pts_all_frames = defaultdict(dict)
            self.obj_pts_all_frames = defaultdict(dict)
            for frame in calib_data:
                self.valid_frames.add(frame)
                for data_type, data in calib_data[frame].items():
                    data_type_lower = data_type.lower()
                    if "object" in data_type_lower:
                        self.obj_pts_all_frames[frame] = data
                    elif "image" in data_type_lower:
                        self.img_pts_all_frames[frame] = data
            for frame in set(self.img_pts_all_frames.keys()) | set(
                self.obj_pts_all_frames.keys()
            ):
                if len(self.img_pts_all_frames[frame]) == len(
                    self.obj_pts_all_frames[frame]
                ):
                    self.valid_frames.add(frame)
        
        # # remove data from frames with odd indices
        # self.valid_frames = [frame for frame in self.valid_frames if frame % 2 == 0]
        # for frame in self.valid_frames:
        #     self.obj_pts_all_frames[frame] = self.obj_pts_all_frames[frame][::2]
        #     self.img_pts_all_frames[frame] = self.img_pts_all_frames[frame][::2]

    def _filter_outliers(self):
        percentile_threshold = 70

        all_errors = []

        for frame_idx in list(self.valid_frames):
            if (
                frame_idx not in self.obj_pts_all_frames
                or frame_idx not in self.img_pts_all_frames
            ):
                continue

            obj_points = np.asarray(
                self.obj_pts_all_frames[frame_idx], dtype=np.float64
            ).reshape(-1, 3)
            img_points = np.asarray(
                self.img_pts_all_frames[frame_idx], dtype=np.float64
            ).reshape(-1, 2)

            if obj_points.shape[0] == 0:
                continue

            camera_matrix = np.array(
                [
                    [self.intrinsics.fx, 0, self.intrinsics.cx],
                    [0, self.intrinsics.fy, self.intrinsics.cy],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )

            dist_coeffs = np.asarray(
                self.intrinsics.dist_coeffs, dtype=np.float64
            ).reshape(-1)

            projected_points, _ = cv2.projectPoints(
                obj_points,
                self.extrinsics.rvec,
                self.extrinsics.tvec,
                camera_matrix,
                dist_coeffs,
            )
            projected_points = projected_points.reshape(-1, 2)

            errors = np.linalg.norm(projected_points - img_points, axis=1)

            for i, error in enumerate(errors):
                all_errors.append((frame_idx, i, error))

        if not all_errors:
            logger.warning("[_filter_outliers] No points to filter")
            return 0

        all_errors.sort(key=lambda x: x[2], reverse=True)

        error_values = [e[2] for e in all_errors]
        threshold = np.percentile(error_values, percentile_threshold)

        points_removed = 0
        frames_modified = set()

        for frame_idx, point_idx, error in all_errors:
            if error > threshold:
                if frame_idx in frames_modified:
                    continue

                obj_pts = self.obj_pts_all_frames[frame_idx]
                img_pts = self.img_pts_all_frames[frame_idx]

                if obj_pts.shape[0] <= point_idx:
                    continue

                self.obj_pts_all_frames[frame_idx] = np.delete(
                    obj_pts, point_idx, axis=0
                )
                self.img_pts_all_frames[frame_idx] = np.delete(
                    img_pts, point_idx, axis=0
                )

                frames_modified.add(frame_idx)
                points_removed += 1

        logger.info(
            f"[_filter_outliers] Removed {points_removed} outlier points with errors > {threshold:.2f} px"
        )
