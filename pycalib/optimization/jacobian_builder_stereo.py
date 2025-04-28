import cv2
import numpy as np

from pycalib.optimization.jacobian_builder_single import JacobianBuilderSingle
from pycalib.optimization.optimizer_configs import setup_logger
from pycalib.optimization.optimizer_single import Optimizer

logger = setup_logger("stereo_jacob_builder")


class JacobianBuilderStereo(JacobianBuilderSingle):
    def __init__(self, optimizer: Optimizer):
        super().__init__()
        self.optimizer = optimizer

    def compute_full_jacobian(self, params):
        """
        Compute the full Jacobian matrix for stereo bundle adjustment.
        """
        self.optimizer._unpack_parameters(params)
        total_params = len(params)
        total_residuals = self._calc_total_residual_num()

        jacobian = np.zeros((total_residuals, total_params), dtype=np.float64)

        param_indices = self._build_param_indices()

        row_idx = 0
        if (
            self.optimizer.all_left_obj_points is not None
            and len(self.optimizer.all_left_obj_points) > 0
        ):
            row_idx = self._add_left_camera_jacobian(jacobian, row_idx, param_indices)

        if (
            self.optimizer.all_right_obj_points is not None
            and len(self.optimizer.all_right_obj_points) > 0
        ):
            expected_rows_after_right = self._calc_total_residual_num()
            if (
                row_idx + 2 * len(self.optimizer.all_right_obj_points)
                > expected_rows_after_right
            ):
                logger.warning(
                    f"Potential row mismatch before adding right jacobian. Current row: {row_idx}, Right points: {len(self.optimizer.all_right_obj_points)}, Expected total rows: {expected_rows_after_right}"
                )

            row_idx = self._add_right_camera_jacobian(jacobian, row_idx, param_indices)

        if row_idx != total_residuals:
            logger.warning(
                f"Final row index {row_idx} does not match total expected residuals {total_residuals}"
            )

        if np.any(np.isnan(jacobian)) or np.any(np.isinf(jacobian)):
            logger.error("NaN/Inf values found in Jacobian matrix.")
            raise ValueError("NaN/Inf values found in Jacobian")

        return jacobian

    def _build_param_indices(self):
        """Build a simple parameter index mapping"""
        indices = {}
        current_idx = 0

        if self.optimizer.left_flags:
            num_left_intr = 0
            if self.optimizer.left_flags.estimate_focal:
                num_left_intr += 2  # fx, fy
            if self.optimizer.left_flags.estimate_principal:
                num_left_intr += 2  # cx, cy
            left_dist_flags = [
                self.optimizer.left_flags.estimate_k1,
                self.optimizer.left_flags.estimate_k2,
                self.optimizer.left_flags.estimate_p1,
                self.optimizer.left_flags.estimate_p2,
                self.optimizer.left_flags.estimate_k3,
            ]
            num_left_intr += sum(left_dist_flags)

            if num_left_intr > 0:
                indices["left_intrinsics"] = slice(
                    current_idx, current_idx + num_left_intr
                )
                current_idx += num_left_intr

        if self.optimizer.right_flags:
            num_right_intr = 0
            if self.optimizer.right_flags.estimate_focal:
                num_right_intr += 2  # fx, fy
            if self.optimizer.right_flags.estimate_principal:
                num_right_intr += 2  # cx, cy
            right_dist_flags = [
                self.optimizer.right_flags.estimate_k1,
                self.optimizer.right_flags.estimate_k2,
                self.optimizer.right_flags.estimate_p1,
                self.optimizer.right_flags.estimate_p2,
                self.optimizer.right_flags.estimate_k3,
            ]
            num_right_intr += sum(right_dist_flags)

            if num_right_intr > 0:
                indices["right_intrinsics"] = slice(
                    current_idx, current_idx + num_right_intr
                )
                current_idx += num_right_intr

        indices["relative_pose"] = slice(current_idx, current_idx + 6)
        current_idx += 6

        indices["left_extrinsics"] = slice(current_idx, current_idx + 6)
        current_idx += 6

        return indices

    def _calc_total_residual_num(self):
        """Calculate the total number of residual elements (2 per point per camera)."""
        num_left_points = (
            self.optimizer.all_left_obj_points.shape[0]
            if self.optimizer.all_left_obj_points is not None
            else 0
        )
        num_right_points = (
            self.optimizer.all_right_obj_points.shape[0]
            if self.optimizer.all_right_obj_points is not None
            else 0
        )
        return 2 * (num_left_points + num_right_points)

    def _add_left_camera_jacobian(self, jacobian, row_idx, param_indices):
        """Add left camera Jacobian blocks to the full Jacobian"""
        if self.optimizer.all_left_obj_points is None:
            logger.warning("Left object points are None in _add_left_camera_jacobian")
            return row_idx

        obj_points = self.optimizer.all_left_obj_points
        num_points = len(obj_points)
        if num_points == 0:
            return row_idx

        camera_matrix = np.array(
            [
                [
                    self.optimizer.left_intrinsics.fx,
                    0,
                    self.optimizer.left_intrinsics.cx,
                ],
                [
                    0,
                    self.optimizer.left_intrinsics.fy,
                    self.optimizer.left_intrinsics.cy,
                ],
                [0, 0, 1],
            ]
        )
        dist_coeffs = np.array(self.optimizer.left_intrinsics.dist_coeffs).reshape(-1)

        rvec = self.optimizer.left_extrinsics.rvec
        tvec = self.optimizer.left_extrinsics.tvec

        if obj_points.ndim != 2 or obj_points.shape[1] != 3:
            raise ValueError(f"Invalid shape for obj_points: {obj_points.shape}")
        if not np.all(np.isfinite(obj_points)):
            raise ValueError("Non-finite values found in obj_points")

        _, cv_jacobian = cv2.projectPoints(
            obj_points, rvec, tvec, camera_matrix, dist_coeffs
        )
        expected_cols = 6
        if "left_intrinsics" in param_indices:
            num_intr = 0
            if self.optimizer.left_flags.estimate_focal:
                num_intr += 2
            if self.optimizer.left_flags.estimate_principal:
                num_intr += 2
            if self.optimizer.left_flags.estimate_k1:
                num_intr += 1
            if self.optimizer.left_flags.estimate_k2:
                num_intr += 1
            if self.optimizer.left_flags.estimate_p1:
                num_intr += 1
            if self.optimizer.left_flags.estimate_p2:
                num_intr += 1
            if self.optimizer.left_flags.estimate_k3:
                num_intr += 1
            expected_cols = 14

        if cv_jacobian.shape != (2 * num_points, expected_cols):
            logger.warning(
                f"Unexpected cv_jacobian shape for left camera. Got {cv_jacobian.shape}, expected ({2 * num_points}, {expected_cols})"
            )

        for pt_idx in range(num_points):
            current_row = row_idx + 2 * pt_idx
            if current_row + 1 >= jacobian.shape[0]:
                logger.error(
                    f"Row index {current_row + 1} out of bounds for Jacobian with shape {jacobian.shape}"
                )
                continue

            if "left_extrinsics" in param_indices:
                extr_slice = param_indices["left_extrinsics"]
                if extr_slice.stop > jacobian.shape[1]:
                    logger.error(
                        f"Extrinsic slice stop index {extr_slice.stop} out of bounds for Jacobian shape {jacobian.shape}"
                    )
                    continue
                if 6 > cv_jacobian.shape[1]:
                    logger.error(
                        f"cv_jacobian shape {cv_jacobian.shape} doesn't have enough columns for extrinsics (needs 6)"
                    )
                    continue

                for j in range(6):
                    jacobian[current_row, extr_slice.start + j] = cv_jacobian[
                        2 * pt_idx, j
                    ]
                    jacobian[current_row + 1, extr_slice.start + j] = cv_jacobian[
                        2 * pt_idx + 1, j
                    ]

            if "left_intrinsics" in param_indices:
                intr_slice = param_indices["left_intrinsics"]
                if intr_slice.stop > jacobian.shape[1]:
                    logger.error(
                        f"Intrinsic slice stop index {intr_slice.stop} out of bounds for Jacobian shape {jacobian.shape}"
                    )
                    continue

                intr_col = 0
                if self.optimizer.left_flags.estimate_focal:
                    jacobian[current_row, intr_slice.start + intr_col] = cv_jacobian[
                        2 * pt_idx, 6
                    ]
                    jacobian[current_row + 1, intr_slice.start + intr_col] = (
                        cv_jacobian[2 * pt_idx + 1, 6]
                    )
                    intr_col += 1

                    jacobian[current_row, intr_slice.start + intr_col] = cv_jacobian[
                        2 * pt_idx, 7
                    ]
                    jacobian[current_row + 1, intr_slice.start + intr_col] = (
                        cv_jacobian[2 * pt_idx + 1, 7]
                    )
                    intr_col += 1

                if self.optimizer.left_flags.estimate_principal:
                    jacobian[current_row, intr_slice.start + intr_col] = cv_jacobian[
                        2 * pt_idx, 8
                    ]
                    jacobian[current_row + 1, intr_slice.start + intr_col] = (
                        cv_jacobian[2 * pt_idx + 1, 8]
                    )
                    intr_col += 1

                    jacobian[current_row, intr_slice.start + intr_col] = cv_jacobian[
                        2 * pt_idx, 9
                    ]
                    jacobian[current_row + 1, intr_slice.start + intr_col] = (
                        cv_jacobian[2 * pt_idx + 1, 9]
                    )
                    intr_col += 1

                if self.optimizer.left_flags.estimate_k1:
                    jacobian[current_row, intr_slice.start + intr_col] = cv_jacobian[
                        2 * pt_idx, 10
                    ]
                    jacobian[current_row + 1, intr_slice.start + intr_col] = (
                        cv_jacobian[2 * pt_idx + 1, 10]
                    )
                    intr_col += 1

                if self.optimizer.left_flags.estimate_k2:
                    jacobian[current_row, intr_slice.start + intr_col] = cv_jacobian[
                        2 * pt_idx, 11
                    ]
                    jacobian[current_row + 1, intr_slice.start + intr_col] = (
                        cv_jacobian[2 * pt_idx + 1, 11]
                    )
                    intr_col += 1

                if self.optimizer.left_flags.estimate_p1:
                    jacobian[current_row, intr_slice.start + intr_col] = cv_jacobian[
                        2 * pt_idx, 12
                    ]
                    jacobian[current_row + 1, intr_slice.start + intr_col] = (
                        cv_jacobian[2 * pt_idx + 1, 12]
                    )
                    intr_col += 1

                if self.optimizer.left_flags.estimate_p2:
                    jacobian[current_row, intr_slice.start + intr_col] = cv_jacobian[
                        2 * pt_idx, 13
                    ]
                    jacobian[current_row + 1, intr_slice.start + intr_col] = (
                        cv_jacobian[2 * pt_idx + 1, 13]
                    )
                    intr_col += 1

                if self.optimizer.left_flags.estimate_k3:
                    jacobian[current_row, intr_slice.start + intr_col] = cv_jacobian[
                        2 * pt_idx, 14
                    ]
                    jacobian[current_row + 1, intr_slice.start + intr_col] = (
                        cv_jacobian[2 * pt_idx + 1, 14]
                    )
                    intr_col += 1

        return row_idx + 2 * num_points

    def _add_right_camera_jacobian(self, jacobian, row_idx, param_indices):
        """Add right camera Jacobian blocks to the full Jacobian"""
        if self.optimizer.all_right_obj_points is None:
            logger.warning("Right object points are None in _add_right_camera_jacobian")
            return row_idx

        obj_points = self.optimizer.all_right_obj_points
        num_points = len(obj_points)
        if num_points == 0:
            return row_idx

        left_rvec = self.optimizer.left_extrinsics.rvec
        left_tvec = self.optimizer.left_extrinsics.tvec
        rel_rvec = self.optimizer.rel_rvec
        rel_tvec = self.optimizer.rel_tvec

        (
            right_rvec,
            right_tvec,
            d_right_rvec_d_left_rvec,
            d_right_rvec_d_left_tvec,
            d_right_tvec_d_left_rvec,
            d_right_tvec_d_left_tvec,
            d_right_rvec_d_rel_rvec,
            d_right_rvec_d_rel_tvec,
            d_right_tvec_d_rel_rvec,
            d_right_tvec_d_rel_tvec,
        ) = self.compose_motion_with_derivatives(
            left_rvec, left_tvec, rel_rvec, rel_tvec
        )

        camera_matrix = np.array(
            [
                [
                    self.optimizer.right_intrinsics.fx,
                    0,
                    self.optimizer.right_intrinsics.cx,
                ],
                [
                    0,
                    self.optimizer.right_intrinsics.fy,
                    self.optimizer.right_intrinsics.cy,
                ],
                [0, 0, 1],
            ]
        )
        dist_coeffs = np.array(self.optimizer.right_intrinsics.dist_coeffs).reshape(-1)

        _, cv_jacobian_full = cv2.projectPoints(
            obj_points, right_rvec, right_tvec, camera_matrix, dist_coeffs
        )

        for pt_idx in range(num_points):
            current_row = row_idx + 2 * pt_idx
            row_slice_cv = slice(2 * pt_idx, 2 * pt_idx + 2)

            d_proj_d_right_rvec = cv_jacobian_full[row_slice_cv, 0:3]
            d_proj_d_right_tvec = cv_jacobian_full[row_slice_cv, 3:6]

            d_proj_d_left_rvec = (
                d_proj_d_right_rvec @ d_right_rvec_d_left_rvec
                + d_proj_d_right_tvec @ d_right_tvec_d_left_rvec
            )

            d_proj_d_left_tvec = (
                d_proj_d_right_rvec @ d_right_rvec_d_left_tvec
                + d_proj_d_right_tvec @ d_right_tvec_d_left_tvec
            )

            d_proj_d_rel_rvec = (
                d_proj_d_right_rvec @ d_right_rvec_d_rel_rvec
                + d_proj_d_right_tvec @ d_right_tvec_d_rel_rvec
            )

            d_proj_d_rel_tvec = (
                d_proj_d_right_rvec @ d_right_rvec_d_rel_tvec
                + d_proj_d_right_tvec @ d_right_tvec_d_rel_tvec
            )

            if "left_extrinsics" in param_indices:
                extr_slice = param_indices["left_extrinsics"]
                jacobian[
                    current_row : current_row + 2,
                    extr_slice.start : extr_slice.start + 3,
                ] = d_proj_d_left_rvec
                jacobian[
                    current_row : current_row + 2,
                    extr_slice.start + 3 : extr_slice.start + 6,
                ] = d_proj_d_left_tvec

            if "relative_pose" in param_indices:
                rel_slice = param_indices["relative_pose"]
                jacobian[
                    current_row : current_row + 2, rel_slice.start : rel_slice.start + 3
                ] = d_proj_d_rel_rvec
                jacobian[
                    current_row : current_row + 2,
                    rel_slice.start + 3 : rel_slice.start + 6,
                ] = d_proj_d_rel_tvec

            if "right_intrinsics" in param_indices:
                intr_slice = param_indices["right_intrinsics"]
                intr_col_target = 0
                cv_col_idx = 6

                if self.optimizer.right_flags.estimate_focal:
                    jacobian[
                        current_row : current_row + 2,
                        intr_slice.start + intr_col_target,
                    ] = cv_jacobian_full[row_slice_cv, cv_col_idx]  # fx
                    intr_col_target += 1
                    cv_col_idx += 1
                    jacobian[
                        current_row : current_row + 2,
                        intr_slice.start + intr_col_target,
                    ] = cv_jacobian_full[row_slice_cv, cv_col_idx]  # fy
                    intr_col_target += 1
                    cv_col_idx += 1

                if self.optimizer.right_flags.estimate_principal:
                    jacobian[
                        current_row : current_row + 2,
                        intr_slice.start + intr_col_target,
                    ] = cv_jacobian_full[row_slice_cv, cv_col_idx]  # cx
                    intr_col_target += 1
                    cv_col_idx += 1
                    jacobian[
                        current_row : current_row + 2,
                        intr_slice.start + intr_col_target,
                    ] = cv_jacobian_full[row_slice_cv, cv_col_idx]  # cy
                    intr_col_target += 1
                    cv_col_idx += 1

                if self.optimizer.right_flags.estimate_k1:
                    jacobian[
                        current_row : current_row + 2,
                        intr_slice.start + intr_col_target,
                    ] = cv_jacobian_full[row_slice_cv, cv_col_idx]
                    intr_col_target += 1
                    cv_col_idx += 1

                if self.optimizer.right_flags.estimate_k2:
                    jacobian[
                        current_row : current_row + 2,
                        intr_slice.start + intr_col_target,
                    ] = cv_jacobian_full[row_slice_cv, cv_col_idx]
                    intr_col_target += 1
                    cv_col_idx += 1

                if self.optimizer.right_flags.estimate_p1:
                    jacobian[
                        current_row : current_row + 2,
                        intr_slice.start + intr_col_target,
                    ] = cv_jacobian_full[row_slice_cv, cv_col_idx]
                    intr_col_target += 1
                    cv_col_idx += 1

                if self.optimizer.right_flags.estimate_p2:
                    jacobian[
                        current_row : current_row + 2,
                        intr_slice.start + intr_col_target,
                    ] = cv_jacobian_full[row_slice_cv, cv_col_idx]
                    intr_col_target += 1
                    cv_col_idx += 1

                if self.optimizer.right_flags.estimate_k3:
                    jacobian[
                        current_row : current_row + 2,
                        intr_slice.start + intr_col_target,
                    ] = cv_jacobian_full[row_slice_cv, cv_col_idx]
                    intr_col_target += 1
                    cv_col_idx += 1

        return row_idx + 2 * num_points

    def compose_motion_with_derivatives(self, left_rvec, left_tvec, rel_rvec, rel_tvec):
        """
        Compute composite motion of two poses with all derivatives.

        Args:
            left_rvec: Left camera rotation vector (3x1 or 3,)
            left_tvec: Left camera translation vector (3x1 or 3,)
            rel_rvec: Relative rotation vector (3x1 or 3,)
            rel_tvec: Relative translation vector (3x1 or 3,)

        Returns:
            tuple: Contains right_rvec, right_tvec, and all 8 Jacobians (3x3 matrices)
                   needed for the chain rule in _process_right_camera_points.
        """
        left_rvec = np.asarray(left_rvec, dtype=np.float64).reshape(3)
        left_tvec = np.asarray(left_tvec, dtype=np.float64).reshape(3)
        rel_rvec = np.asarray(rel_rvec, dtype=np.float64).reshape(3)
        rel_tvec = np.asarray(rel_tvec, dtype=np.float64).reshape(3)

        R_left, dR_left_d_left_rvec = cv2.Rodrigues(left_rvec)
        R_rel, dR_rel_d_rel_rvec = cv2.Rodrigues(rel_rvec)

        if dR_left_d_left_rvec.shape == (3, 9):
            dR_left_d_left_rvec = dR_left_d_left_rvec.T
        elif dR_left_d_left_rvec.shape != (9, 3):
            raise ValueError(
                f"[JacobianBuilderStereo.compose_motion_with_derivatives] Unexpected left Rodrigues Jacobian shape {dR_left_d_left_rvec.shape}. Expecting (9, 3) or (3, 9)."
            )

        if dR_rel_d_rel_rvec.shape == (3, 9):
            dR_rel_d_rel_rvec = dR_rel_d_rel_rvec.T
        elif dR_rel_d_rel_rvec.shape != (9, 3):
            raise ValueError(
                f"[JacobianBuilderStereo.compose_motion_with_derivatives] Unexpected relative Rodrigues Jacobian shape {dR_rel_d_rel_rvec.shape}. Expecting (9, 3) or (3, 9)."
            )

        R_right = R_rel @ R_left
        right_tvec = R_rel @ left_tvec + rel_tvec
        right_rvec, d_right_rvec_d_R_right_flat = cv2.Rodrigues(R_right)

        if d_right_rvec_d_R_right_flat.shape == (9, 3):
            d_right_rvec_d_R_right_flat = d_right_rvec_d_R_right_flat.T
        elif d_right_rvec_d_R_right_flat.shape != (3, 9):
            raise ValueError(
                f"[JacobianBuilderStereo.compose_motion_with_derivatives] Unexpected R_right -> rvec Jacobian shape {d_right_rvec_d_R_right_flat.shape}. Expecting (3, 9) or (9, 3)."
            )

        dR_right_flat_d_left_rvec = np.zeros((9, 3), dtype=np.float64)
        for j in range(3):
            dR_left_j = dR_left_d_left_rvec[:, j].reshape(3, 3)
            dR_right_j = R_rel @ dR_left_j
            dR_right_flat_d_left_rvec[:, j] = dR_right_j.flatten()

        d_right_rvec_d_left_rvec = (
            d_right_rvec_d_R_right_flat @ dR_right_flat_d_left_rvec
        )

        dR_right_flat_d_rel_rvec = np.zeros((9, 3), dtype=np.float64)
        for j in range(3):
            dR_rel_j = dR_rel_d_rel_rvec[:, j].reshape(3, 3)
            dR_right_j = dR_rel_j @ R_left
            dR_right_flat_d_rel_rvec[:, j] = dR_right_j.flatten()

        d_right_rvec_d_rel_rvec = d_right_rvec_d_R_right_flat @ dR_right_flat_d_rel_rvec

        d_right_tvec_d_rel_rvec = np.zeros((3, 3), dtype=np.float64)
        left_tvec_col = left_tvec.reshape(3, 1)
        for j in range(3):
            dR_rel_j = dR_rel_d_rel_rvec[:, j].reshape(3, 3)
            d_right_tvec_d_rel_rvec[:, j] = (dR_rel_j @ left_tvec_col).flatten()

        d_right_rvec_d_left_tvec = np.zeros((3, 3), dtype=np.float64)
        d_right_rvec_d_rel_tvec = np.zeros((3, 3), dtype=np.float64)
        d_right_tvec_d_left_rvec = np.zeros((3, 3), dtype=np.float64)
        d_right_tvec_d_left_tvec = R_rel.copy()
        d_right_tvec_d_rel_tvec = np.eye(3, dtype=np.float64)

        return (
            right_rvec.reshape(3, 1),
            right_tvec.reshape(3, 1),
            d_right_rvec_d_left_rvec,
            d_right_rvec_d_left_tvec,
            d_right_tvec_d_left_rvec,
            d_right_tvec_d_left_tvec,
            d_right_rvec_d_rel_rvec,
            d_right_rvec_d_rel_tvec,
            d_right_tvec_d_rel_rvec,
            d_right_tvec_d_rel_tvec,
        )
