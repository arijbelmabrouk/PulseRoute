import cv2
import time
from collections import deque

def initialize_camera(camera_index=0):
    """
    Opens camera, warms it up, measures actual fps.
    camera_index=0 means first available camera.
    Returns cap and actual_fps or None, None if failed.
    """
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("ERROR: No camera detected")
        return None, None

    # Warm up — discard first 30 frames
    print("Warming up camera...")
    for i in range(30):
        cap.read()

    # Measure actual fps over 5 seconds
    print("Measuring camera fps...")
    frame_count = 0
    start_time = time.time()
    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if ret:
            frame_count += 1

    actual_fps = frame_count / 5

    # Get resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Camera ready: {width}x{height} @ {actual_fps:.1f}fps")

    # Classify camera quality
    if actual_fps < 15:
        print("WARNING: FPS too low for reliable measurement")
        print("Minimum required: 15fps")
    elif actual_fps < 25:
        print("ACCEPTABLE: Signal quality may be reduced")
    else:
        print("GOOD: Camera suitable for full pipeline")

    # Warn about low resolution
    if width < 320 or height < 240:
        print("WARNING: Resolution too low for reliable ROI extraction")
    elif width < 640 or height < 480:
        print("ACCEPTABLE: Resolution may affect ROI quality")
    else:
        print(f"GOOD: Resolution sufficient for ROI extraction")

    return cap, actual_fps


def stream_frames(cap, actual_fps):
    """
    Streams frames continuously.
    Converts BGR to RGB.
    Calculates rolling fps dynamically.
    Returns final rolling fps.
    """
    # Adaptive rolling window — always represents 1 second
    window_seconds = 1
    maxlen = max(int(actual_fps * window_seconds), 2)
    timestamps = deque(maxlen=maxlen)

    rolling_fps = actual_fps  # start with measured fps as fallback
    frame_rgb = None

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame")
                break

            # Record timestamp
            timestamps.append(time.time())

            # Calculate rolling fps
            if len(timestamps) >= 2:
                rolling_fps = (len(timestamps) - 1) / (
                    timestamps[-1] - timestamps[0]
                )

            # Convert BGR to RGB — must happen before any processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display feed — use original BGR frame for display
            display = frame.copy()
            cv2.putText(
                display,
                f"FPS: {rolling_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            cv2.putText(
                display,
                f"Resolution: {frame.shape[1]}x{frame.shape[0]}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            cv2.imshow("Camera Feed", display)

            # Detect significant fps drop during session
            if rolling_fps < 15:
                print(f"WARNING: FPS dropped to {rolling_fps:.1f}"
                      f" — signal quality affected")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("Stream stopped by user")

    finally:
        # Always release camera even if error occurs
        cap.release()
        cv2.destroyAllWindows()
        print(f"Session ended. Final FPS: {rolling_fps:.1f}")

    return rolling_fps


# Entry point
if __name__ == "__main__":
    cap, actual_fps = initialize_camera(camera_index=0)
    if cap is not None:
        final_fps = stream_frames(cap, actual_fps)