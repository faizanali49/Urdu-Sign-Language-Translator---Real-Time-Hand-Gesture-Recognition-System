# collection_continuous_data.py
import cv2
import os
import time
import platform # For better path handling

# --- Configuration ---
DATASET_DIR = "continuous_dataset"
VIDEO_DIR = os.path.join(DATASET_DIR, "videos")
DEFAULT_FPS = 20.0 # Fallback FPS if camera doesn't provide it
VIDEO_CODEC = 'MJPG' # More compatible than XVID on some systems

# --- Initialization ---
os.makedirs(VIDEO_DIR, exist_ok=True)
print(f"Dataset directory ensured at: {os.path.abspath(VIDEO_DIR)}")

cap = cv2.VideoCapture(0) # Use default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get camera properties (try to get actual FPS and Resolution)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    print(f"Warning: Could not get camera FPS. Using default: {DEFAULT_FPS}")
    fps = DEFAULT_FPS

print(f"Camera resolution: {width}x{height}, FPS: {fps:.2f}")
fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)

print("\n=== Continuous Data Collection ===")
print("1. Enter a descriptive name for the continuous gesture (e.g., 'swipe_left', 'wave').")
print("2. Enter the number of times you want to record this gesture.")
print("3. When prompted, press 'r' to start recording.")
print("4. Perform the gesture clearly.")
print("5. Press 'q' to stop recording the current clip.")
print("6. Enter 'exit' as the gesture name to quit.")
print("===================================\n")

# --- Main Loop ---
while True:
    gesture_name = input("Enter gesture name (or 'exit'): ").strip().lower().replace(" ", "_") # Sanitize name
    if gesture_name == "exit":
        break
    if not gesture_name:
        print("Gesture name cannot be empty.")
        continue

    # Create directory for the specific gesture
    gesture_path = os.path.join(VIDEO_DIR, gesture_name)
    os.makedirs(gesture_path, exist_ok=True)

    try:
        num_recordings = int(input(f"How many recordings for '{gesture_name}'? "))
        if num_recordings <= 0:
            print("Please enter a positive number.")
            continue
    except ValueError:
        print("Invalid number.")
        continue

    print("-" * 20)

    for i in range(num_recordings):
        print(f"\nPrepare for Recording {i+1}/{num_recordings} for '{gesture_name}'.")
        print("Press 'r' when ready to start recording.")

        # Wait for 'r' key press
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Dropped frame while waiting.")
                time.sleep(0.05)
                continue
            frame = cv2.flip(frame, 1) # Horizontal flip
            # Display instructions
            cv2.putText(frame, f"Press 'r' to record '{gesture_name}' ({i+1}/{num_recordings})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Collecting Continuous Data", frame)
            if cv2.waitKey(1) & 0xFF == ord('r'):
                break

        # --- Start Recording ---
        # Use timestamp for unique filenames
        timestamp = int(time.time())
        # Use OS-specific path separator if needed, though os.path.join handles it
        filename = os.path.join(gesture_path, f"{gesture_name}_{timestamp}.avi")

        # Create VideoWriter object for this recording
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        if not out.isOpened():
            print(f"Error: Could not open VideoWriter for '{filename}'. Check codec/permissions.")
            continue # Skip to next recording iteration

        print(f"🔴 Recording '{filename}'... Press 'q' to stop.")
        recording_start_time = time.time()
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Dropped frame during recording.")
                continue

            frame = cv2.flip(frame, 1)
            out.write(frame) # Write frame to file
            frame_count += 1

            # Display recording status
            elapsed_time = time.time() - recording_start_time
            cv2.putText(frame, f"REC ● {elapsed_time:.1f}s", (width - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Collecting Continuous Data", frame)

            # Check for 'q' key press to stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"\nStopping recording {i+1}.")
                break

        # --- Stop Recording ---
        out.release() # Finalize the video file
        recording_duration = time.time() - recording_start_time
        print(f"✅ Recording saved: {filename} ({recording_duration:.2f}s, {frame_count} frames)")

    print("-" * 20)

# --- Cleanup ---
print("Continuous data collection complete.")
cap.release()
cv2.destroyAllWindows()