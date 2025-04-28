import multiprocessing
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Union

import numpy as np
from splib.calibration import stereoCalibration
from splib.utils import configLoader
from splib.utils.resourceManager import ResourceManager

from pycalib.feature_processing.feature_extractor import (
    FeatureAligner,
    ImageProcessor,
)
from pycalib.feature_processing.feature_processing_utils import (
    polyfit,
    filter_features,
    save_cam_proj_feat_vis,
)
from pycalib.feature_processing.feature_extractor_cv import FeatureExtractorCV


def load_config(cfg_file="debugging.yaml"):
    cfg_path = os.path.join("pycalib/configs", cfg_file)
    configLoader.loadProjectConfig(cfg_path)
    resource_manager = ResourceManager.getInstance()
    CalShift = configLoader.initMeasurementClass(resource_manager["config"])
    return CalShift


def load_images(folder_path):
    """
    Load reference and calibration images from folder structure.

    Args:
        folder_path: Path to folders containing images

    Returns:
        Tuple of (ref_images_dict, calib_images_dict) where:
        - ref_images_dict: Dictionary mapping folder indices to reference images
        - calib_images_dict: Dictionary mapping folder indices to lists of calibration images
    """
    folder_path = Path(folder_path)
    ref_images_dict = {}
    calib_images_dict = {}

    # Pre-sort folders once
    folders = sorted(
        folder_path.iterdir(),
        key=lambda x: int(x.stem) if x.stem.isdigit() else -1,
    )

    # Process each folder sequentially
    for folder in folders:
        folder_stem = folder.stem
        folder_idx = int(folder_stem)

        # Load images
        img_files = sorted(
            folder.iterdir(),
            key=lambda x: int(x.stem.replace("img", ""))
            if x.stem.startswith("img")
            else -1,
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

        # Store the images
        ref_images_dict[folder_idx] = folder_ref_image
        calib_images_dict[folder_idx] = folder_calib_images

    return ref_images_dict, calib_images_dict


def extract_cam_features(ref_images_dict):
    """
    Extract and track features from camera images without filtering.

    Args:
        ref_images_dict: Dictionary mapping folder indices to reference images

    Returns:
        Camera feature data dictionary
        2d-3d correspondences for the camera features
    """
    # Define cache file paths
    original_cam_feat_file = Path("pycalib/data/cache/cam_feature_data.pkl")
    filtered_cam_feat_file = Path("pycalib/data/cache/filtered_cam_feature_data.pkl")

    # Check if both cache files exist
    if original_cam_feat_file.exists() and filtered_cam_feat_file.exists():
        print("Loading camera feature data from existing data...")
        with open(original_cam_feat_file, "rb") as infile:
            cam_feature_data = pickle.load(infile)
        with open(filtered_cam_feat_file, "rb") as infile:
            filtered_cam_feature_data = pickle.load(infile)
    else:
        print(
            "No cache files not found, extracting features..."
        )  # Updated print message
        # Pre-allocate data structures
        ref_image_dict = {}
        feature_dict = {}
        ref_point_dict = {}  # Initialize dictionary for reference points
        cam_feature_data = defaultdict(dict)
        filtered_cam_feature_data = defaultdict(dict)

        # Process each folder sequentially using the ref images
        for frame_id, ref_image in sorted(ref_images_dict.items()):
            extractor = FeatureExtractorCV()

            current_features = extractor.detect_grid_features(ref_image)

            ref_image_dict[frame_id] = ref_image

            if current_features is not None and len(current_features) > 0:
                # Store features for this specific frame_id
                feature_dict[frame_id] = current_features
                # Calculate the center of the features
                center = np.mean(current_features, axis=0)
                # Calculate distances from each feature to the center
                distances = np.linalg.norm(current_features - center, axis=1)
                # Find the index of the feature closest to the center
                closest_idx = np.argmin(distances)
                # Select the closest feature as the reference point
                ref_point_dict[frame_id] = current_features[closest_idx]
            else:
                raise ValueError(f"No features detected in frame {frame_id}.")

        # Align 2d-3d points
        aligner = FeatureAligner()
        # Iterate through frames that have features and a reference point
        for frame_id, current_frame_features in sorted(feature_dict.items()):
            # Check if features were detected and a reference point exists for this frame
            if frame_id in ref_point_dict:
                tracked_point = ref_point_dict[frame_id]

                # Pass the correct features and the calculated reference point
                img_points, obj_points, corres = aligner._align_points_to_grid(
                    current_frame_features, tracked_point, frame_id
                )

                # Only store if alignment produced valid points
                if img_points.size > 0 and obj_points.size > 0:
                    cam_feature_data[frame_id] = {
                        "ref_image": ref_image_dict[frame_id],
                        "imagePoints": img_points,
                        "objectPoints": obj_points,
                        "2d_3d_corres": corres,
                    }

                    # Filter features
                    filtered_img_points, filtered_obj_points = filter_features(
                        cam_name="Camera",
                        folder_idx=frame_id,
                        img_points=img_points,
                        obj_points=obj_points,
                    )
                    filtered_cam_feature_data[frame_id] = {
                        "ref_image": ref_image_dict[frame_id],
                        "imagePoints": filtered_img_points,
                        "objectPoints": filtered_obj_points,
                        "2d_3d_corres": corres,
                    }
                # Removed the 'else raise ValueError' as frames without successful alignment might be skipped
            else:
                # This case handles frames where no features were detected initially
                print(
                    f"Skipping alignment for frame {frame_id} due to missing features or reference point."
                )

        # Save the extracted and filtered data
        if cam_feature_data:
            with open(original_cam_feat_file, "wb") as outfile:
                pickle.dump(
                    dict(cam_feature_data), outfile
                )  # Convert back to dict for saving
        else:
            print("Warning: No valid feature data generated after alignment.")

        if filtered_cam_feature_data:
            with open(filtered_cam_feat_file, "wb") as outfile:
                pickle.dump(
                    dict(filtered_cam_feature_data), outfile
                )  # Convert back to dict for saving
        else:
            print("Warning: No valid filtered feature data generated after alignment.")

    return cam_feature_data, filtered_cam_feature_data


def process_folder_polyfit(
    folder_idx,
    cam_feature_data,
    calib_images_dict,
    CalShift,
    radius=50,
    dense_thresh=0.5,
    max_threads_per_polyfit=2,
):
    """
    Process a single folder with polyfit for projector feature extraction.

    Args:
        folder_idx: Index of the folder to process
        cam_feature_data: Camera feature data dictionary
        calib_images_dict: Dictionary of calibration images
        CalShift: Instance of CalibrationShift class
        radius: Radius for polynomial fitting
        dense_thresh: Density threshold for polynomial fitting
        max_threads_per_polyfit: Maximum number of threads to use for polyfit

    Returns:
        Tuple of (folder_idx, fitted_map, masked_img_points, masked_obj_points)
    """
    data = cam_feature_data[folder_idx]
    img_points = data["imagePoints"]
    obj_points = data["objectPoints"]
    calib_images = calib_images_dict[folder_idx]

    # Get projector map - moved from extract_cam_features
    CalShift.deleteImageStack()
    CalShift.addImages(calib_images)
    proj_map = CalShift.getProjectorMap(calib_images)

    # Apply polynomial fitting to projector map (this internally uses parallelism)
    fitted_map, feature_mask = polyfit(
        proj_map,
        img_points,
        radius=radius,
        dense_thresh=dense_thresh,
        max_workers=max_threads_per_polyfit,
    )

    # Apply feature mask
    masked_img_points = img_points[feature_mask]
    masked_obj_points = obj_points[feature_mask]

    return folder_idx, fitted_map, masked_img_points, masked_obj_points


def extract_proj_features(
    cam_feature_data,
    CalShift,
    calib_images_dict,
    radius=50,
    dense_thresh=0.5,
):
    """
    Extract projector features using polyfit filtering on camera features.

    Args:
        cam_feature_data: Camera feature data from extract_cam_features
        CalShift: Instance of CalibrationShift class
        calib_images_dict: Dictionary mapping folder indices to lists of calibration images

    Returns:
        Tuple of projector feature data, and filtered camera feature data
    """
    # Determine optimal number of workers based on CPU count
    try:
        cpu_count = multiprocessing.cpu_count()
    except NotImplementedError:
        cpu_count = 2
    max_threads_per_polyfit = min(2, cpu_count)

    proj_K_guess = np.array(
        [
            [1300.0, 0.0, 384.0],
            [0.0, 1300.0, 384.0],
            [0.0, 0.0, 1.0],
        ]
    )
    proj_distort_guess = np.zeros((5, 1))

    proj_features = Path("pycalib/data/cache/proj_feature_data.pkl")
    masked_features = Path("pycalib/data/cache/masked_cam_feature_data.pkl")
    filtered_proj_features = Path("pycalib/data/cache/filtered_proj_feature_data.pkl")

    if (
        proj_features.exists()
        and masked_features.exists()
        and filtered_proj_features.exists()
    ):
        print("Loading projector feature data from exsisting data...")
        with open(proj_features, "rb") as infile:
            proj_feature_data = pickle.load(infile)
        with open(masked_features, "rb") as infile:
            masked_cam_feature_data = pickle.load(infile)
        with open(filtered_proj_features, "rb") as infile:
            filtered_proj_feature_data = pickle.load(infile)
    else:
        print("No existing projector feature data found, extracting features...")
        masked_cam_feature_data = defaultdict(dict)

        # Process folders in parallel for polyfit operation
        folder_indices = list(cam_feature_data.keys())

        # Process folders sequentially, but polyfit internally is parallel
        for folder_idx in folder_indices:
            folder_idx, fitted_map, masked_img_points, masked_obj_points = (
                process_folder_polyfit(
                    folder_idx,
                    cam_feature_data,
                    calib_images_dict,
                    CalShift,
                    radius,
                    dense_thresh,
                    max_threads_per_polyfit,
                )
            )

            # Store filtered results
            masked_cam_feature_data[folder_idx] = {
                "projectorMap": fitted_map,
                "imagePoints": masked_img_points,
                "objectPoints": masked_obj_points,
                "img": cam_feature_data[folder_idx]["ref_image"],
            }

        # Convert to list format for stereoCalibration.get_projector_features
        cam_feature_list = []
        for idx, data in masked_cam_feature_data.items():
            cam_feature_list[idx] = data

        temp_data, _, _ = stereoCalibration.get_projector_features(
            feature_data=cam_feature_list,
            projector_resolution=(768, 768),
            pro_K_guess=proj_K_guess,
            pro_dist_guess=proj_distort_guess,
            removal_type="overall",
            removal_algorithm="3sigma",
        )

        # Convert list to defaultdict format
        proj_feature_data = defaultdict(dict)
        filtered_proj_feature_data = defaultdict(dict)
        for idx in range(len(temp_data)):
            if temp_data[idx] == -1:
                continue
            proj_feature_data[idx] = {
                "imagePoints": temp_data[idx]["imagePoints"],
                "objectPoints": temp_data[idx]["objectPoints"],
            }

            # Filter projector features
            filtered_img_points, filtered_obj_points = filter_features(
                cam_name="Projector",
                folder_idx=idx,
                img_points=temp_data[idx]["imagePoints"],
                obj_points=temp_data[idx]["objectPoints"],
            )
            filtered_proj_feature_data[idx] = {
                "imagePoints": filtered_img_points,
                "objectPoints": filtered_obj_points,
            }

        # Save both processed data files
        with open(proj_features, "wb") as outfile:
            pickle.dump((proj_feature_data), outfile)
        with open(masked_features, "wb") as outfile:
            pickle.dump(masked_cam_feature_data, outfile)
        with open(filtered_proj_features, "wb") as outfile:
            pickle.dump(filtered_proj_feature_data, outfile)

    return proj_feature_data, masked_cam_feature_data, filtered_proj_feature_data


def extract_features(
    image_folders: Union[str, Path],
    save_plots_for_debugging=True,
):
    """
    Perform the complete camera-projector calibration pipeline.
    """
    image_folders = Path(image_folders)

    if not image_folders.exists():
        raise FileNotFoundError(f"Path does not exist: {image_folders}")
    if not image_folders.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {image_folders}")

    # Load configuration
    CalShift = load_config()

    ref_images_dict, calib_images_dict = load_images(image_folders)

    # Extract camera features
    cam_feature_data, filtered_cam_feature_data = extract_cam_features(ref_images_dict)

    # Extract projector features
    proj_feature_data, masked_cam_feature_data, filtered_proj_feature_data = (
        extract_proj_features(
            cam_feature_data,
            CalShift,
            calib_images_dict,
            radius=50,
            dense_thresh=0.5,
        )
    )

    if save_plots_for_debugging:
        save_cam_proj_feat_vis(
            cam_feature_data,
            proj_feature_data,
        )


if __name__ == "__main__":
    extract_features(
        "pycalib/data/image_sources/image_data_cv",
        save_plots_for_debugging=True,
    )
