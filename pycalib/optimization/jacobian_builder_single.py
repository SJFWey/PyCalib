import cv2
import numpy as np

from pycalib.optimization.optimizer_configs import (
    Intrinsics,
    Extrinsics,
    OptimizerFlags,
    setup_logger,
)

logger = setup_logger("single_jacob_builder")


class JacobianBuilderSingle:
    def __init__(self):
        pass

    def compute_jacobian(
        self,
        all_points_3d: np.ndarray,
        extrinsics: Extrinsics,
        intrinsics: Intrinsics,
        flags: OptimizerFlags,
        unpack_parameters_func,
        params=None,
    ):
        """
        Compute the full Jacobian matrix for bundle adjustment.
        """
        if params is not None:
            unpack_parameters_func(params)

        if all_points_3d is None or len(all_points_3d) == 0:
            logger.error("[compute_jacobian] No 3D points provided.")
            raise ValueError("No 3D points provided to compute Jacobian.")

        num_total_residuals = 2 * len(all_points_3d)

        rvec = np.asarray(extrinsics.rvec, dtype=np.float64).reshape(
            3,
        )
        tvec = np.asarray(extrinsics.tvec, dtype=np.float64).reshape(
            3,
        )
        camera_matrix = np.array(
            [
                [intrinsics.fx, 0, intrinsics.cx],
                [0, intrinsics.fy, intrinsics.cy],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.array(intrinsics.dist_coeffs, dtype=np.float64).reshape(-1)

        num_total_params = 0
        if flags.estimate_extrinsics:
            num_total_params += 6
        if flags.estimate_focal:
            num_total_params += 2
        if flags.estimate_principal:
            num_total_params += 2
        if flags.estimate_k1:
            num_total_params += 1
        if flags.estimate_k2:
            num_total_params += 1
        if flags.estimate_p1:
            num_total_params += 1
        if flags.estimate_p2:
            num_total_params += 1
        if flags.estimate_k3:
            num_total_params += 1

        _, cv_jacobian = cv2.projectPoints(
            all_points_3d, rvec, tvec, camera_matrix, dist_coeffs
        )

        jacobian = np.zeros((num_total_residuals, num_total_params))

        col_idx = 0
        if flags.estimate_extrinsics:
            jacobian[:, col_idx : col_idx + 6] = cv_jacobian[:, 0:6]
            col_idx += 6

        if flags.estimate_focal:
            jacobian[:, col_idx] = cv_jacobian[:, 6]
            col_idx += 1
            jacobian[:, col_idx] = cv_jacobian[:, 7]
            col_idx += 1

        if flags.estimate_principal:
            jacobian[:, col_idx] = cv_jacobian[:, 8]
            col_idx += 1
            jacobian[:, col_idx] = cv_jacobian[:, 9]
            col_idx += 1

        cv_dist_idx = 10
        if flags.estimate_k1:
            jacobian[:, col_idx] = cv_jacobian[:, cv_dist_idx]
            col_idx += 1
        cv_dist_idx += 1
        if flags.estimate_k2:
            jacobian[:, col_idx] = cv_jacobian[:, cv_dist_idx]
            col_idx += 1
        cv_dist_idx += 1
        if flags.estimate_p1:
            jacobian[:, col_idx] = cv_jacobian[:, cv_dist_idx]
            col_idx += 1
        cv_dist_idx += 1
        if flags.estimate_p2:
            jacobian[:, col_idx] = cv_jacobian[:, cv_dist_idx]
            col_idx += 1
        cv_dist_idx += 1
        if flags.estimate_k3:
            jacobian[:, col_idx] = cv_jacobian[:, cv_dist_idx]
            col_idx += 1

        if col_idx != num_total_params:
            logger.error(
                f"Jacobian construction error: Expected {num_total_params} columns, built {col_idx} columns."
            )
            raise ValueError("Jacobian column count mismatch during construction.")

        return jacobian

    def validate_params_consistency(
        self,
        pack_func,
        unpack_func,
    ):
        params = pack_func()
        params_copy = params.copy()

        unpack_func(params_copy)
        repacked_params = pack_func()

        if np.allclose(params, repacked_params, rtol=1e-5, atol=1e-8):
            return True

        diff_indices = np.where(
            ~np.isclose(params, repacked_params, rtol=1e-5, atol=1e-8)
        )[0]
        max_diff = np.max(np.abs(params - repacked_params))

        diff_info = ", ".join(
            [
                f"param[{idx}]: orig={params[idx]:.6f}, repacked={repacked_params[idx]:.6f}"
                for idx in diff_indices[:5]
            ]
        )

        error_msg = (
            f"Parameter packing/unpacking inconsistency detected: max diff={max_diff:.6f}, "
            f"{len(diff_indices)}/{len(params)} parameters inconsistent. "
            f"Examples: {diff_info}"
            f"{' (and more...)' if len(diff_indices) > 10 else ''}"
        )

        logger.error(
            "[JacobianBuilderSingle.validate_parameter_consistency] %s", error_msg
        )

        logger.error(
            "Parameter shapes: original=%s, repacked=%s",
            params.shape,
            repacked_params.shape,
        )

        param_order_info = "Parameter ordering check: "
        if len(params) == len(repacked_params):
            # Check first few and last few parameters to identify structure issues
            n_check = min(3, len(params))
            param_order_info += (
                f"First {n_check}: orig={params[:n_check]}, repacked={repacked_params[:n_check]}; "
                f"Last {n_check}: orig={params[-n_check:]}, repacked={repacked_params[-n_check:]}"
            )
        else:
            param_order_info += (
                f"Length mismatch: orig={len(params)}, repacked={len(repacked_params)}"
            )

        logger.error(param_order_info)

        raise ValueError(error_msg)
