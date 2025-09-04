# utils.py
import numpy as np
import json
import os

def normalize_landmarks(landmarks_sequence_raw):
    """
    Normalizes a sequence of landmarks relative to the wrist (landmark 0)
    of the first frame where a hand is present. Handles frames with no hand (zeros).

    Args:
        landmarks_sequence_raw (np.ndarray): Shape (num_frames, 63) or similar.
                                                Assumes 0s represent no hand detected.

    Returns:
        np.ndarray: Normalized landmark sequence, same shape.
                    Returns None if input is invalid or no hand is ever detected.
    """
    if landmarks_sequence_raw is None or landmarks_sequence_raw.ndim != 2 or landmarks_sequence_raw.shape[1] != 63:
        print("Warning: Invalid input shape for normalization.")
        return None

    normalized_sequence = np.copy(landmarks_sequence_raw).astype(np.float32) # Work on a copy
    first_hand_frame_idx = -1

    # Find the first frame with actual hand data (non-zero wrist coordinates)
    for i in range(len(normalized_sequence)):
        # Check if wrist position (first 3 values) is non-zero
        if np.any(normalized_sequence[i, :3] != 0):
            first_hand_frame_idx = i
            break

    if first_hand_frame_idx == -1:
        # No hand detected in the entire sequence, return zeros or None?
        # Returning zeros matches the placeholder, but signifies no actual data.
        # print("Warning: No hand detected in sequence for normalization reference.")
        return normalized_sequence # Return the original zeros

    # Get reference wrist position from the first valid frame
    ref_wrist_x = normalized_sequence[first_hand_frame_idx, 0]
    ref_wrist_y = normalized_sequence[first_hand_frame_idx, 1]
    ref_wrist_z = normalized_sequence[first_hand_frame_idx, 2]

    # Normalize all frames based on the reference wrist
    for i in range(len(normalized_sequence)):
        # Only normalize if hand data is present (non-zero wrist) in that frame
        if np.any(normalized_sequence[i, :3] != 0):
            for j in range(0, 63, 3): # Iterate through x, y, z of each landmark
                normalized_sequence[i, j]   = normalized_sequence[i, j]   - ref_wrist_x
                normalized_sequence[i, j+1] = normalized_sequence[i, j+1] - ref_wrist_y
                normalized_sequence[i, j+2] = normalized_sequence[i, j+2] - ref_wrist_z
        # Else: Keep the frame as zeros (placeholder for no hand)

    return normalized_sequence


def load_class_names(path="training/class_labels_continuous.json"):
    """Loads the class names list from the JSON file."""
    try:
        with open(path, 'r', encoding='utf-8') as f: # Specify encoding
            class_names = json.load(f)
        return class_names
    except FileNotFoundError:
        print(f"Error: Class labels file not found at {path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {path}")
        return None

def save_class_names(class_names, path="training/class_labels_continuous.json"):
    """Saves the class names list to a JSON file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True) # Ensure directory exists
        with open(path, 'w', encoding='utf-8') as f: # Specify encoding
            json.dump(class_names, f, indent=4) # Use indent for readability
        print(f"Class labels saved to {path}")
    except Exception as e:
        print(f"Error saving class labels to {path}: {e}")