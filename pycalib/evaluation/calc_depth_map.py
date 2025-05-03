import json
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from splib.utils import configLoader
from splib.utils.resourceManager import ResourceManager

from pycalib.evaluation.evaluator import evaluate
from pycalib.feature_processing.feature_extractor import ImageProcessor
from pycalib.optimization.optimizer_configs import Extrinsics, Intrinsics


def load_config(cfg_file="height_map_calculating.yaml"):
    cfg_path = os.path.join("pycalib/configs", cfg_file)
    configLoader.loadProjectConfig(cfg_path)
    resource_manager = ResourceManager.getInstance()
    measure = configLoader.initMeasurementClass(resource_manager["config"])
    return measure


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


def generate_params_file(
    cam_params_pkl: str = "pycalib/data/cache/calib_result_Camera.pkl",
    proj_params_pkl: str = "pycalib/data/cache/calib_result_Projector.pkl",
    output_json: str = "pycalib/configs/cam_params_depth_map.json",
):
    cam_params_path = Path(cam_params_pkl)
    proj_params_path = Path(proj_params_pkl)
    output_path = Path(output_json)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cam_params_path, "rb") as f:
        cam_calib_result = pickle.load(f)

    with open(proj_params_path, "rb") as f:
        proj_calib_result = pickle.load(f)

    cam_params = extract_device_params(cam_calib_result)
    proj_params = extract_device_params(proj_calib_result)

    params_data = {
        "cams": {"cam": cam_params},
        "projectors": {"projector": proj_params},
    }

    with open(output_path, "w") as f:
        json.dump(params_data, f, indent=4)


def load_images(folder_path: str):
    ref_image = None
    img_list = []
    folder_path = Path(folder_path)
    all_img_files = [
        f
        for f in folder_path.iterdir()
        if f.suffix == ".npy" and f.stem.startswith("img")
    ]

    ref_image_path = None
    other_img_files = []
    for f in all_img_files:
        if f.name == "img0.npy":
            ref_image_path = f
        else:
            index = int(f.stem[3:])
            if index > 0:
                other_img_files.append(f)

    if ref_image_path:
        ref_image = ImageProcessor.load_image(ref_image_path)
    else:
        print("Warning: img0.npy not found in the specified folder.")

    other_img_files.sort(key=lambda x: int(x.stem[3:]))

    for img_file in other_img_files:
        img = ImageProcessor.load_image(img_file)
        img_list.append(img)

    output_path = Path("pycalib/results/debugging/height_map_debugging/original_images")
    output_path.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path / "img0.png"), ref_image)
    for i, img in enumerate(img_list):
        img_path = output_path / f"img{i + 1}.png"
        cv2.imwrite(str(img_path), img)

    return ref_image, img_list


if __name__ == "__main__":
    # params_file = Path("pycalib/configs/cam_params_depth_map.json")
    # params_file_cv = Path("pycalib/configs/cam_params_depth_map_cv.json")
    # if not params_file.exists():
    #     generate_params_file()
    # if not params_file_cv.exists():
    #     generate_params_file(
    #         cam_params_pkl="pycalib/data/cache/calib_result_Camera_cv.pkl",
    #         proj_params_pkl="pycalib/data/cache/calib_result_Projector_cv.pkl",
    #         output_json="pycalib/configs/cam_params_depth_map_cv.json",
    #     )

    # ref_image, img_list = load_images("pycalib/data/image_sources/image_data_freq25/18")
    ref_image, img_list = load_images("pycalib/data/image_sources/images_sphere")

    cfg_file = "height_map_calculating.yaml"
    measure = load_config(cfg_file)
    measure.deleteImageStack()
    measure.addImages(img_list)

    xyz_map = measure.getHeightmap()

    heightmap_z = xyz_map[:, :, 2]

    plt.subplot(1, 2, 1)
    plt.imshow(ref_image)
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(heightmap_z, cmap="viridis")
    plt.colorbar()
    plt.title("Height Map (Z)")
    plt.savefig(
        "pycalib/results/debugging/height_map_debugging/heightmap_z.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()

    xyz = measure.getMesh()[0]
    np.save("pycalib/data/cache/hole10.npy", xyz)
    # measure.saveMesh("pycalib/data/cache/")

    evaluate()
