import pickle
from pathlib import Path

import cv2
import numpy as np

from pycalib.optimization.optimizer_configs import ParamsGuess
from pycalib.optimization.optimizer_single import Optimizer


def calib_single_cv(
    cam_name: str,
    feature_data: dict[dict],
    intris_guess: ParamsGuess = None,
):
    optimizer = Optimizer(
        cam_name,
        feature_data,
        intris_guess,
    )
    optimizer._load_feature_data(feature_data)
    optimizer._pack_all_points()

    intrinsics = optimizer.intrinsics

    camera_matrix = np.array(
        [
            [intrinsics.fx, 0, intrinsics.cx],
            [0, intrinsics.fy, intrinsics.cy],
            [0, 0, 1],
        ],
    )
    
    obj_pts = [optimizer.all_obj_points]
    img_pts = [optimizer.all_img_points]

    error, camera_matrix, dist_coeffs, rvec, tvec = cv2.calibrateCamera(
        obj_pts,
        img_pts,
        optimizer.image_size,
        camera_matrix,
        intrinsics.dist_coeffs,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS,
    )
    
    return error, camera_matrix, dist_coeffs, rvec, tvec


if __name__ == "__main__":
    use_filtered_data = True
    if use_filtered_data:
        cam_features = Path("pycalib/data/cache/filtered_proj_feature_data.pkl")
    else:
        cam_features = Path("pycalib/data/cache/proj_feature_data.pkl")

    if not cam_features.exists():
        raise FileNotFoundError("Feature data file not found")

    with open(cam_features, "rb") as infile:
        cam_feature_data = pickle.load(infile)

    intris_guess_cam = ParamsGuess(
        image_size=(1280, 720),
        fx=1000,
        fy=1000,
        cx=640,
        cy=360,
    )

    error, camera_matrix, dist_coeffs, rvec, tvec = calib_single_cv(
        "Camera",
        cam_feature_data,
        intris_guess=intris_guess_cam,
    )
    
    print(f"Camera matrix:\n{np.array2string(camera_matrix, precision=2, suppress_small=True)}")
    print(f"Distortion coefficients:\n{np.array2string(dist_coeffs, precision=4, suppress_small=True)}")
    print(f"Error: {error}")
    print(f"rvec:\n{np.array(rvec).flatten()}")
    print(f"tvec:\n{np.array(tvec).flatten()}")
