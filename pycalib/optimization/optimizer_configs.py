import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class OptimizerFlags:
    """Flags controlling which parameters to optimize."""

    estimate_focal: bool = True
    estimate_principal: bool = True

    estimate_k1: bool = True
    estimate_k2: bool = True
    estimate_p1: bool = True
    estimate_p2: bool = True
    estimate_k3: bool = True

    estimate_extrinsics: bool = True

    def to_dict(self):
        return {
            "focal": self.estimate_focal,
            "principal_point": self.estimate_principal,
            "distortion": [
                self.estimate_k1,
                self.estimate_k2,
                self.estimate_p1,
                self.estimate_p2,
                self.estimate_k3,
            ],
            "extrinsics": self.estimate_extrinsics,
        }


@dataclass
class StereoCalibrationFlags:
    """Flags controlling which parameters to optimize in stereo calibration."""

    def __init__(
        self,
        estimate_left_intrinsics: bool = True,
        estimate_right_intrinsics: bool = True,
    ):
        if estimate_left_intrinsics:
            self.left_flags = OptimizerFlags(
                estimate_focal=True,
                estimate_principal=True,
                estimate_k1=True,
                estimate_k2=True,
                estimate_p1=True,
                estimate_p2=True,
                estimate_k3=True,
                estimate_extrinsics=True,
            )
        else:
            self.left_flags = OptimizerFlags(
                estimate_focal=False,
                estimate_principal=False,
                estimate_k1=False,
                estimate_k2=False,
                estimate_p1=False,
                estimate_p2=False,
                estimate_k3=False,
                estimate_extrinsics=True,
            )

        if estimate_right_intrinsics:
            self.right_flags = OptimizerFlags(
                estimate_focal=True,
                estimate_principal=True,
                estimate_k1=True,
                estimate_k2=True,
                estimate_p1=True,
                estimate_p2=True,
                estimate_k3=True,
                estimate_extrinsics=True,
            )
        else:
            self.right_flags = OptimizerFlags(
                estimate_focal=False,
                estimate_principal=False,
                estimate_k1=False,
                estimate_k2=False,
                estimate_p1=False,
                estimate_p2=False,
                estimate_k3=False,
                estimate_extrinsics=True,
            )


@dataclass
class OptimizerParams:
    """Configuration parameters for the optimization process.

    Attributes:
        max_iter: Maximum number of iterations
        verbose: Verbosity level
        opt_method: Optimization method
        ftol: Tolerance for termination by the change of the cost function
        xtol: Tolerance for termination by the change of the independent variables
        gtol: Tolerance for termination by the norm of the gradient
        refine_extrinsics: Whether to refine extrinsics per frame in each iteration
    """

    def __init__(
        self,
        max_iter=100,
        verbose=1,
        opt_method="lm",
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
    ):
        self.max_iter = max_iter
        self.verbose = verbose
        self.opt_method = opt_method
        self.ftol = ftol
        self.xtol = xtol
        self.gtol = gtol


@dataclass
class OptimizationState:
    """Tracks the state of the optimization process."""

    params_history: list[np.ndarray] = field(default_factory=list)
    current_iteration: int = 0
    current_error: float = float("inf")
    best_error: float = float("inf")
    best_params: np.ndarray | None = None
    initial_cost: float | None = None
    final_result: Any | None = None
    cost_history: list[float] = field(default_factory=list)

    def update(self, params: np.ndarray, error: float) -> None:
        self.params_history.append(params.copy())

        self.current_iteration += 1
        self.current_error = error

        if error < self.best_error:
            self.best_error = error
            self.best_params = params.copy()


@dataclass
class ParamsGuess:
    """
    Initialize camera intrinsic parameters.
    """

    def __init__(
        self,
        image_size: tuple[int, int] | None = None,
        fx: float | None = None,
        fy: float | None = None,
        cx: float | None = None,
        cy: float | None = None,
        skew: float | None = None,
        dist_coeffs: np.ndarray | None = None,
    ):
        self.image_size = image_size if image_size else None

        self.fx = fx
        self.fy = fy

        self.cx = cx if cx is not None else (image_size[0] / 2 if image_size else None)
        self.cy = cy if cy is not None else (image_size[1] / 2 if image_size else None)
        self.skew = 0.0 if skew is None else skew

        if dist_coeffs is None:
            self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            self.dist_coeffs = dist_coeffs


@dataclass
class Intrinsics:
    """Stores intrinsic camera parameters."""

    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    dist_coeffs: np.ndarray = field(
        default_factory=lambda: np.zeros(5, dtype=np.float64)
    )

    def __post_init__(self):
        if not isinstance(self.dist_coeffs, np.ndarray):
            self.dist_coeffs = np.array(self.dist_coeffs, dtype=np.float64)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

    def to_dict(self):
        """Converts Intrinsics object to a dictionary."""
        k_matrix = [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "dist_coeffs": self.dist_coeffs.tolist(),
            "K": k_matrix,
        }


@dataclass
class Extrinsics:
    rvec: np.ndarray
    tvec: np.ndarray

    def to_dict(self):
        return {
            "rvec": self.rvec.tolist(),
            "tvec": self.tvec.tolist(),
        }


def setup_logger(
    logger_name: str = None,
    log_level: int = logging.DEBUG,
    log_file: str = None,
    cam_name: str = None,
    log_dir: str = None,
) -> logging.Logger:
    """
    Setup and configure a logger with optional camera-specific log file.

    Args:
        logger_name: Name of the logger
        log_level: Logging level (default: logging.INFO)
        log_file: Path to log file (if None, uses default or camera-specific path)
        cam_name: Camera name for specific log file (if None, uses logger_name)
        log_dir: Directory to store log files (if None, uses default)

    Returns:
        Configured logger instance
    """
    # Get or create logger
    logger_name = f"{cam_name}_calibration" if cam_name else logger_name
    logger = logging.getLogger(logger_name)

    # If logger already has handlers and is configured, return it
    if logger.handlers and logger.level != logging.NOTSET:
        return logger

    logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicate logging
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Setup log directory
    if log_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(script_dir, "..", "..", "logs")

    os.makedirs(log_dir, exist_ok=True)

    # Determine log file path
    if log_file is None:
        if cam_name:
            log_file = os.path.join(log_dir, f"{cam_name}_calibration.log")
        else:
            log_file = os.path.join(log_dir, f"{logger_name}.log")

    # Create file handler
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(log_level)

    # Set formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)

    # Add console handler if needed
    if not any(
        isinstance(handler, logging.StreamHandler) for handler in logger.handlers
    ):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Color formatter for console output
        class ColorFormatter(logging.Formatter):
            YELLOW = "\033[33m"
            RED = "\033[31m"
            RESET = "\033[0m"

            def format(self, record):
                message = super().format(record)
                if record.levelno == logging.WARNING:
                    message = f"{self.YELLOW}{message}{self.RESET}"
                elif record.levelno >= logging.ERROR:
                    message = f"{self.RED}{message}{self.RESET}"
                return message

        color_formatter = ColorFormatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(color_formatter)
        logger.addHandler(console_handler)

    return logger
