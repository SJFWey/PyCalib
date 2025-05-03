# File: export_features_to_matlab.py
import pickle
from pathlib import Path
import numpy as np
from scipy.io import savemat
import os  # Added os import, needed if parent.mkdir used

# Define paths
input_pkl_path = Path("calibpy/data/cache/filtered_cam_feature_data.pkl")
output_mat_path = Path(
    "C:/Users/xjwei/OneDrive/Masterarbeit/Calib/Tests/feature_data/cam_features_from_py.mat"
)


# --- Preprocessing function (adapted from test_single.py) ---
def preprocess_feature_data(feature_data):
    """
    Extracts image and object points from the loaded data structure.
    Handles both {'image_points': {frame: pts}, 'object_points': {frame: pts}} and
    {frame: {'imagePoints': pts, 'objectPoints': pts}} structures.

    Returns:
        Tuple[dict, dict, list]: Dictionaries for image points, object points (keyed by frame_id),
                                 and a list of valid frame IDs.
    """
    image_points = {}
    object_points = {}
    valid_frames = []

    if (
        isinstance(feature_data, dict)
        and "image_points" in feature_data
        and "object_points" in feature_data
    ):
        # Structure is {'image_points': {frame_id: pts}, 'object_points': {frame_id: pts}}
        temp_img_pts = feature_data["image_points"]
        temp_obj_pts = feature_data["object_points"]
        all_frame_ids = set(temp_img_pts.keys()) | set(temp_obj_pts.keys())
        for frame_id in sorted(list(all_frame_ids)):
            if frame_id in temp_img_pts and frame_id in temp_obj_pts:
                img_pts = np.asarray(temp_img_pts[frame_id])
                obj_pts = np.asarray(temp_obj_pts[frame_id])
                # Check for non-empty and matching number of points
                if img_pts.shape[0] > 0 and img_pts.shape[0] == obj_pts.shape[0]:
                    image_points[frame_id] = img_pts
                    object_points[frame_id] = obj_pts
                    valid_frames.append(frame_id)
                else:
                    print(
                        f"Warning: Skipping frame {frame_id} due to mismatched or empty points."
                    )
            else:
                print(
                    f"Warning: Skipping frame {frame_id} due to missing image or object points."
                )

    elif isinstance(feature_data, dict):
        # Try the older structure {frame_id: {'imagePoints': pts, 'objectPoints': pts}}
        for frame_id in sorted(feature_data.keys()):
            data = feature_data[frame_id]
            if (
                isinstance(data, dict)
                and "imagePoints" in data
                and "objectPoints" in data
            ):
                img_pts = np.asarray(data["imagePoints"])
                obj_pts = np.asarray(data["objectPoints"])
                # Check for non-empty and matching number of points
                if img_pts.shape[0] > 0 and img_pts.shape[0] == obj_pts.shape[0]:
                    image_points[frame_id] = img_pts
                    object_points[frame_id] = obj_pts
                    valid_frames.append(frame_id)
                else:
                    print(
                        f"Warning: Skipping frame {frame_id} due to mismatched or empty points."
                    )
            else:
                print(
                    f"Warning: Skipping frame {frame_id} due to missing 'imagePoints' or 'objectPoints' key, or unexpected data format."
                )
    else:
        raise TypeError("Loaded data structure is not a recognized dictionary format.")

    if not valid_frames:
        print("Warning: No valid frames found after preprocessing.")

    return image_points, object_points, valid_frames


# --- Main script ---

# Check if input file exists
if not input_pkl_path.exists():
    raise FileNotFoundError(f"Input file not found: {input_pkl_path}")

# Load data from pickle file
print(f"Loading data from {input_pkl_path}...")
try:
    with open(input_pkl_path, "rb") as infile:
        raw_proj_feature_data = pickle.load(infile)
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading pickle file: {e}")
    exit()

# Preprocess the data
print("Preprocessing feature data...")
try:
    image_points_all_frames, object_points_all_frames, valid_frames = (
        preprocess_feature_data(raw_proj_feature_data)
    )
except TypeError as e:
    print(f"Error during preprocessing: {e}")
    exit()

# Prepare data for MATLAB - Accumulate points and indices from all frames
all_x_points = []
all_X_points = []
all_plane_indices = []  # List to store frame indices for each point
total_points_processed = 0

print(f"Processing {len(valid_frames)} valid frames for MATLAB export...")
for i, frame_id in enumerate(valid_frames):
    img_pts = image_points_all_frames[frame_id]  # Shape (n, 2)
    obj_pts = object_points_all_frames[frame_id]  # Shape (n, 3)

    # Basic validation of point shapes
    if img_pts.ndim != 2 or img_pts.shape[1] != 2:
        print(
            f"Warning: Skipping frame {frame_id} due to unexpected image points shape {img_pts.shape}"
        )
        continue
    if obj_pts.ndim != 2 or obj_pts.shape[1] != 3:
        print(
            f"Warning: Skipping frame {frame_id} due to unexpected object points shape {obj_pts.shape}"
        )
        continue
    if img_pts.shape[0] != obj_pts.shape[0]:
        # This should have been caught by preprocessing, but double-check
        print(
            f"Warning: Skipping frame {frame_id} due to inconsistent point counts ({img_pts.shape[0]} vs {obj_pts.shape[0]})"
        )
        continue
    num_points_in_frame = img_pts.shape[0]
    if num_points_in_frame == 0:
        print(f"Info: Skipping frame {frame_id} as it contains zero points.")
        continue

    # Transpose points: image points to (2, n), object points to (3, n)
    x_i = img_pts.T
    X_i = obj_pts.T

    # Add to the lists for later concatenation
    all_x_points.append(x_i)
    all_X_points.append(X_i)

    # Create and store plane indices (1-based, as double) for this frame
    plane_indices_for_frame = np.full(
        (1, num_points_in_frame), i + 1, dtype=np.float64
    )  # Changed dtype to float64
    all_plane_indices.append(plane_indices_for_frame)

    total_points_processed += num_points_in_frame
    # print(f"  Processed frame {frame_id} -> added {num_points_in_frame} points") # Optional: more verbose logging

# Combine all points and indices into single arrays
matlab_data = {}
if all_x_points and all_X_points and all_plane_indices:
    x_1 = np.hstack(all_x_points)  # Resulting shape (2, N)
    X_1 = np.hstack(all_X_points)  # Resulting shape (3, N)
    plane_index = np.hstack(all_plane_indices)  # Resulting shape (1, N), now float64

    # Assign the combined arrays to matlab_data
    matlab_data["x_1"] = x_1
    matlab_data["X_1"] = X_1
    matlab_data["plane_index"] = plane_index
    print(
        f"Combined data: x_1 shape {x_1.shape}, X_1 shape {X_1.shape}, plane_index shape {plane_index.shape} (dtype: {plane_index.dtype}) ({total_points_processed} total points)"
    )  # Added dtype to printout
else:
    print("No valid points found across all frames to combine.")


# Save data to .mat file
if matlab_data:
    print(f"Saving combined data to {output_mat_path}...")
    try:
        # Ensure the output directory exists
        output_mat_path.parent.mkdir(parents=True, exist_ok=True)
        # Use do_compression=True for potentially smaller files
        savemat(str(output_mat_path), matlab_data, do_compression=True)
        print("MATLAB file saved successfully.")
    except Exception as e:
        print(f"Error saving MATLAB file: {e}")
else:
    print("No valid data was processed to save to MATLAB file.")
