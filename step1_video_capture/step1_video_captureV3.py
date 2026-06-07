import cv2
import time
import numpy as np
from collections import deque


# ─────────────────────────────────────────
# Pre-flight validation constants
# ─────────────────────────────────────────
MIN_FPS          = 15       # below this, HR timing is unreliable
MIN_WIDTH        = 320      # below this, ROI extraction fails
MIN_HEIGHT       = 240
PREFLIGHT_FRAMES = 10       # frames to sample for black/corrupt check
BLACK_THRESHOLD  = 10       # mean pixel value below this = black frame
CORRUPT_MAX_STD  = 2.0      # std below this across all channels = frozen/corrupt


def _check_frame_quality(cap):
    """
    Grab PREFLIGHT_FRAMES frames and check for:
      1. All-black frames (camera covered, hardware fault)
      2. Frozen/corrupt frames (std near zero = same frame repeated)
      3. Consistent valid signal (at least 80% of frames must pass)

    Returns (ok: bool, reason: str)
    """
    results = []

    for _ in range(PREFLIGHT_FRAMES):
        ret, frame = cap.read()
        if not ret:
            results.append(("fail", "cap.read() returned False"))
            continue

        mean_val = float(np.mean(frame))
        std_val  = float(np.std(frame))

        if mean_val < BLACK_THRESHOLD:
            results.append(("black", f"mean={mean_val:.1f}"))
        elif std_val < CORRUPT_MAX_STD:
            results.append(("frozen", f"std={std_val:.2f}"))
        else:
            results.append(("ok", ""))

    statuses   = [r[0] for r in results]
    ok_count   = statuses.count("ok")
    pass_rate  = ok_count / len(results)

    if pass_rate < 0.8:
        # Majority of frames failed — report the dominant problem
        if statuses.count("black") >= statuses.count("frozen"):
            return False, (
                "Camera is producing black frames. "
                "Check that the lens is not covered and the camera is "
                "not disabled in privacy settings."
            )
        elif statuses.count("frozen") > 0:
            return False, (
                "Camera is producing frozen or corrupt frames. "
                "Try unplugging and reconnecting the camera."
            )
        else:
            return False, (
                "Camera failed to deliver frames (cap.read() returned False). "
                "Check USB connection or camera permissions."
            )

    return True, ""


def initialize_camera(camera_index=0):
    """
    Opens camera, runs pre-flight validation, warms it up,
    measures actual fps.

    Pre-flight checks (NEW):
      - Frame delivery  : cap.read() must succeed
      - Black frames    : mean pixel value must be above threshold
      - Frozen frames   : std must show real variation (not a stuck sensor)
      - Resolution      : must be >= 320x240
      - FPS             : must be >= 15 after measurement

    Returns (cap, actual_fps) on success, (None, None) on any failure.
    All failures print a clear human-readable message before returning.
    """
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # ── 1. Basic open check ───────────────────────────────────────
    if not cap.isOpened():
        print("ERROR: No camera detected at index", camera_index)
        print("       Check that the camera is connected and not in use "
              "by another application.")
        return None, None

    # ── 2. Resolution check ───────────────────────────────────────
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width < MIN_WIDTH or height < MIN_HEIGHT:
        print(f"ERROR: Camera resolution {width}x{height} is too low.")
        print(f"       Minimum required: {MIN_WIDTH}x{MIN_HEIGHT}.")
        print("       ROI extraction will fail at this resolution.")
        cap.release()
        return None, None

    # ── 3. Frame quality pre-flight ───────────────────────────────
    # Run BEFORE the warm-up discard so we catch hardware faults early.
    print("Running pre-flight camera check...")
    quality_ok, quality_reason = _check_frame_quality(cap)

    if not quality_ok:
        print(f"ERROR: Pre-flight failed — {quality_reason}")
        cap.release()
        return None, None

    print("Pre-flight passed.")

    # ── 4. Warm up — discard first 30 frames ─────────────────────
    print("Warming up camera...")
    for _ in range(30):
        cap.read()

    # ── 5. Measure actual fps over 5 seconds ──────────────────────
    print("Measuring camera fps...")
    frame_count = 0
    start_time  = time.time()

    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if ret:
            frame_count += 1

    actual_fps = frame_count / 5

    # ── 6. FPS gate ───────────────────────────────────────────────
    if actual_fps < MIN_FPS:
        print(f"ERROR: Measured fps is {actual_fps:.1f}, "
              f"minimum required is {MIN_FPS}.")
        print("       Heart rate timing will be unreliable at this fps.")
        print("       Try closing other applications or improving lighting.")
        cap.release()
        return None, None

    # ── 7. Quality classification ─────────────────────────────────
    print(f"Camera ready: {width}x{height} @ {actual_fps:.1f}fps")

    if actual_fps < 25:
        print("ACCEPTABLE: FPS is low — signal quality may be reduced")
    else:
        print("GOOD: Camera suitable for full pipeline")

    if width < 640 or height < 480:
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
        print("Stream stopped by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Session ended. Final FPS: {rolling_fps:.1f}")

    return rolling_fps


# Entry point
if __name__ == "__main__":
    cap, actual_fps = initialize_camera(camera_index=0)
    if cap is not None:
        final_fps = stream_frames(cap, actual_fps)