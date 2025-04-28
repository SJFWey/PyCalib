from typing import Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


class PointTracker:
    """
    Tracks a single reference point across a sequence of images using
    Lucas-Kanade (LK) optical flow, with refinement based on feature detection.
    """

    def __init__(
        self,
        win_size: Tuple[int, int] = (31, 31),
        max_level: int = 4,
        criteria: Tuple[int, int, float] = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            50,
            0.001,
        ),
        refine_thresh: Optional[float] = None,
    ):
        """
        Initializes the Lucas-Kanade Optical Flow tracker.

        Args:
            win_size: Size of the search window at each pyramid level for LK.
            max_level: 0-based maximal pyramid level number for LK.
            criteria: Termination criteria (type, max_iter, epsilon) for the
                      iterative search algorithm in LK.
            refine_thresh: Optional maximum distance (in pixels) between
                                  the LK prediction and a detected feature for
                                  refinement. If the nearest detected feature is
                                  further than this threshold, refinement fails
                                  for that frame. If None, the closest feature is always used.
        """
        self.lk_params = dict(winSize=win_size, maxLevel=max_level, criteria=criteria)
        self.refine_thresh = refine_thresh

    def _find_nearest_feature(
        self, pred_pt: np.ndarray, feat_pts: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Finds the feature point closest to the target point within an optional threshold.

        Args:
            pred_pt: The (x, y) coordinates to search around (shape (2,)).
            feat_pts: Array of detected feature points (N, 2).

        Returns:
            The coordinates of the nearest feature point (shape (2,)), or None
            if no features are provided or the nearest feature is beyond the
            refinement_threshold.
        """
        if not len(feat_pts):
            return None

        dists = np.linalg.norm(feat_pts - pred_pt, axis=1)
        nearest_idx = np.argmin(dists)
        min_dist = dists[nearest_idx]

        if self.refine_thresh is not None and min_dist > self.refine_thresh:
            return None

        return feat_pts[nearest_idx]

    def _find_init_ref_point(
        self, feats: np.ndarray, first_img: np.ndarray
    ) -> np.ndarray:
        """
        Allows the user to manually select the initial reference point from the
        detected features in the first image.

        Args:
            feats: Detected features in the first frame (N, 2).
            first_img: The first image frame (used for display).

        Returns:
            The coordinates (x, y) of the selected initial reference point.

        Raises:
            ValueError: If no features are detected or if selection is cancelled.
        """
        if feats is None or len(feats) == 0:
            raise ValueError(
                "No features detected in the first frame to select an initial reference point."
            )

        # Use the manual selection method
        print("Please select the initial reference point in the displayed window.")
        sel_pt = self.select_reference_point(first_img, feats)
        return sel_pt

    def select_reference_point(self, img: np.ndarray, feats: np.ndarray) -> np.ndarray:
        """
        Allow manual selection of the reference point by clicking on the image.

        Args:
            img: Input image as numpy array (uint8). Should be displayable
                   by matplotlib (e.g., grayscale or RGB).
            feats: Array of detected feature points (N, 2) in the image.

        Returns:
            Coordinates (x, y) of the detected feature point nearest to the click.

        Raises:
            ValueError: If no features are provided or if manual selection times out
                        or is cancelled without a valid click.
        """
        if feats is None or len(feats) == 0:
            raise ValueError("Cannot select reference point: No features provided.")

        ref_pt = None

        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray")  # Assume grayscale, adjust if needed
        ax.scatter(
            feats[:, 0], feats[:, 1], c="yellow", s=10, label="Detected Features"
        )
        ax.set_title("Choose a desired reference point (Close window to cancel)")
        ax.legend()
        plt.draw()  # Initial draw

        # Use a blocking call to wait for click
        clicked = plt.ginput(1, timeout=-1)  # timeout=-1 waits indefinitely

        if clicked:
            click_pt = np.array(clicked[0])
            ref_pt = self._find_nearest_feature(click_pt, feats)

            if ref_pt is not None:
                print(f"Selected point nearest click: {ref_pt}")
                ax.plot(
                    ref_pt[0],
                    ref_pt[1],
                    "rx",
                    markersize=10,
                    label="Selected Point",
                )
                ax.legend()
                plt.draw()
                plt.pause(1.5)  # Show selection briefly
            else:
                # This case should ideally not happen if features exist,
                # unless refine_thresh is very strict, but handle it.
                plt.close(fig)
                raise ValueError(
                    "Selection failed: No feature found near the clicked point "
                    f"(check refine_thresh if set: {self.refine_thresh})."
                )

        plt.close(fig)

        if ref_pt is None:
            # This happens if the user closed the window before clicking
            raise ValueError("Reference point selection cancelled or failed.")

        return ref_pt

    def track(
        self,
        img_dict: Dict[int, np.ndarray],
        feat_dict: Dict[int, Optional[np.ndarray]],
        init_ref_pt: Optional[np.ndarray] = None,
    ) -> Dict[int, Optional[np.ndarray]]:
        """
        Tracks a reference point through a sequence of images using LK Optical Flow,
        refining the tracked point using pre-computed features.

        Args:
            img_dict: Dictionary mapping frame indices to grayscale images (numpy arrays).
                         The dictionary should be sortable by frame index.
            feat_dict: Dictionary mapping frame indices to pre-detected
                           feature coordinates (N, 2 numpy array) or None if
                           detection failed for a frame. Must contain entries
                           for all frames present in img_dict.
            init_ref_pt: Optional. The (x, y) coordinates (shape (2,)
                                      or (1, 2)) of the reference point in the
                                      first frame. If None, the user will be prompted
                                      to manually select the point from the first frame's
                                      features.

        Returns:
            Dictionary mapping frame indices to the tracked and refined
            reference point coordinates (np.ndarray shape (2,)).
            The value is None if tracking or refinement failed for that frame.
        """
        # Define blur parameters (kernel size) - adjust if needed
        blur_ksize = (5, 5)

        if not feat_dict:
            raise ValueError("feat_dict cannot be empty.")

        try:
            sorted_frames = sorted(img_dict.items())
            frame_ids = [item[0] for item in sorted_frames]
            imgs = [item[1] for item in sorted_frames]

            if not all(idx in feat_dict for idx in frame_ids):
                missing = [idx for idx in frame_ids if idx not in feat_dict]
                raise ValueError(f"Missing feature data for frame indices: {missing}")

            tracked_pts: Dict[int, Optional[np.ndarray]] = {}
            prev_pt: Optional[np.ndarray] = None

            # Initialize first frame
            first_id = frame_ids[0]
            first_img = imgs[0]  # Original first image

            if first_img is None or len(first_img.shape) != 2:
                return {idx: None for idx in frame_ids}

            # --- Preprocess first image for LK ---
            processed_first_img = cv2.GaussianBlur(first_img, blur_ksize, 0)
            prev_gray = processed_first_img.copy()
            # --- End Preprocessing ---

            feats_first = feat_dict.get(first_id)

            if feats_first is None or not len(feats_first):
                raise ValueError(f"No features for first frame ({first_id})")

            # Determine the initial reference point
            if init_ref_pt is not None:
                init_pt_flat = np.array(init_ref_pt).flatten()
                if init_pt_flat.shape != (2,):
                    raise ValueError("init_ref_pt must be convertible to shape (2,)")
                # Refine user-provided point to the nearest detected feature
                refined_init = self._find_nearest_feature(init_pt_flat, feats_first)
                if refined_init is None:
                    thresh_info = (
                        f" (threshold: {self.refine_thresh})"
                        if self.refine_thresh is not None
                        else ""
                    )
                    raise ValueError(
                        f"Provided init_ref_pt {init_pt_flat} "
                        f"is not close enough to any detected feature{thresh_info}."
                    )
                start_pt = refined_init
            else:
                # Manually select the initial reference point using the original first image
                start_pt = self._find_init_ref_point(
                    feats_first,
                    first_img,  # Use original image for display
                )

            tracked_pts[first_id] = start_pt.astype(np.float32)  # Store initial point
            # Prepare point for LK: needs shape (1, 1, 2) and float32 type
            prev_pt = start_pt.reshape(1, 1, 2).astype(np.float32)

            # --- Tracking Loop (Frames > 0) ---
            for i in range(1, len(imgs)):
                curr_id = frame_ids[i]
                curr_img = imgs[i]  # Original current image

                # Handle missing images in the sequence
                if curr_img is None:
                    print(
                        f"Warning: Image for frame {curr_id} is missing. Marking as untracked."
                    )
                    tracked_pts[curr_id] = None
                    prev_pt = None  # Tracking chain is broken
                    continue

                # --- Preprocess current image for LK ---
                processed_curr_img = cv2.GaussianBlur(curr_img, blur_ksize, 0)
                curr_gray = processed_curr_img.copy()
                # --- End Preprocessing ---

                # If tracking was lost in a previous step, mark current as untracked
                if prev_pt is None:
                    tracked_pts[curr_id] = None
                    prev_gray = (
                        curr_gray.copy()
                    )  # Update prev_gray for the *next* iteration
                    continue

                # Apply LK Optical Flow
                next_pt, status, err = cv2.calcOpticalFlowPyrLK(
                    prev_gray, curr_gray, prev_pt, None, **self.lk_params
                )

                # Check tracking status
                lk_ok = status is not None and status[0][0] == 1

                refined_pt: Optional[np.ndarray] = None
                if lk_ok:
                    pred_pt = next_pt[0, 0, :]  # Shape (2,)

                    # --- Refinement Step ---
                    curr_feats = feat_dict.get(curr_id)

                    if curr_feats is not None and len(curr_feats) > 0:
                        refined_pt = self._find_nearest_feature(pred_pt, curr_feats)
                        if refined_pt is None:
                            print(
                                f"Warning: Refinement failed for frame {curr_id}. "
                                f"Nearest feature too far from LK estimate ({pred_pt}). "
                                f"Marking as untracked."
                            )
                    else:
                        print(
                            f"Warning: No pre-computed features found for frame {curr_id}. "
                            f"Cannot refine LK estimate. Marking as untracked."
                        )
                        refined_pt = None

                else:  # LK failed
                    print(
                        f"Warning: LK tracking failed directly for frame {curr_id}. Marking as untracked."
                    )
                    refined_pt = None

                tracked_pts[curr_id] = (
                    refined_pt.astype(np.float32) if refined_pt is not None else None
                )

                # Update state for the next iteration
                prev_gray = (
                    curr_gray.copy()
                )  # Use the processed image for the next step
                prev_pt = (
                    refined_pt.reshape(1, 1, 2).astype(np.float32)
                    if refined_pt is not None
                    else None
                )

        except Exception as e:
            print(f"An error occurred during tracking: {e}")
            if "frame_ids" in locals():
                processed = set(tracked_pts.keys())
                for idx in frame_ids:
                    if idx not in processed:
                        tracked_pts[idx] = None
            return tracked_pts  # Return potentially partial results

        return tracked_pts
