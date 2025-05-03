import pickle
from pathlib import Path

import cv2
import numpy as np

from pycalib.optimization.optimizer_configs import ParamsGuess, Extrinsics, Intrinsics
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

    results = {
        "cam_name": cam_name,
        "image_size": intris_guess.image_size,
        "error": error,
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "rvec": rvec,
        "tvec": tvec,
    }
    return results


def save_results(results: dict, file_path: str):
    reorgnized_results = {}
    reorgnized_results["cam_name"] = results["cam_name"]
    reorgnized_results["image_size"] = results["image_size"]
    reorgnized_results["error"] = results["error"]

    if results["rvec"] and results["tvec"]:
        reorgnized_results["extrinsics"] = Extrinsics(
            np.array(results["rvec"]).flatten(), np.array(results["tvec"]).flatten()
        )

    camera_matrix = np.array(results["camera_matrix"])
    reorgnized_results["intrinsics"] = Intrinsics(
        fx=camera_matrix[0][0],
        fy=camera_matrix[1][1],
        cx=camera_matrix[0][2],
        cy=camera_matrix[1][2],
        dist_coeffs=np.array(results["dist_coeffs"]).flatten(),
    )

    with open(file_path, "wb") as outfile:
        pickle.dump(reorgnized_results, outfile)


if __name__ == "__main__":
    use_filtered_data = True
    if use_filtered_data:
        cam_features = Path("pycalib/data/cache/filtered_cam_feature_data.pkl")
    else:
        cam_features = Path("pycalib/data/cache/cam_feature_data.pkl")

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

    results = calib_single_cv(
        "Camera",
        cam_feature_data,
        intris_guess=intris_guess_cam,
    )

    save_results(results, "pycalib/data/cache/calib_result_Camera_cv.pkl")

    print(
        f"Camera matrix:\n{np.array2string(results['camera_matrix'], precision=2, suppress_small=True)}"
    )
    print(
        f"Distortion coefficients:\n{np.array2string(results['dist_coeffs'], precision=4, suppress_small=True)}"
    )
    print(f"Error: {results['error']}")
    print(f"rvec:\n{np.array(results['rvec']).flatten()}")
    print(f"tvec:\n{np.array(results['tvec']).flatten()}")

    #############################################################

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

    results = calib_single_cv(
        "Projector",
        cam_feature_data,
        intris_guess=intris_guess_cam,
    )

    save_results(results, "pycalib/data/cache/calib_result_Projector_cv.pkl")

    print(
        f"Camera matrix:\n{np.array2string(results['camera_matrix'], precision=2, suppress_small=True)}"
    )
    print(
        f"Distortion coefficients:\n{np.array2string(results['dist_coeffs'], precision=4, suppress_small=True)}"
    )
    print(f"Error: {results['error']}")
    print(f"rvec:\n{np.array(results['rvec']).flatten()}")
    print(f"tvec:\n{np.array(results['tvec']).flatten()}")

