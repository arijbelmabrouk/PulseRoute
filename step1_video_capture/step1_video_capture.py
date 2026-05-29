import cv2
import time

def initialize_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("ERROR: No camera detected")
        return None, None

    # Measure actual fps for 3 seconds
    frame_count = 0
    start_time = time.time()
    while time.time() - start_time < 3:
        ret, frame = cap.read()
        if ret:
            frame_count += 1

    actual_fps = frame_count / 3
    print(f"Camera ready: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {actual_fps:.1f}fps")

    if actual_fps < 15:
        print("WARNING: FPS too low for reliable measurement")
    elif actual_fps < 25:
        print("ACCEPTABLE: Reduced signal quality")
    else:
        print("GOOD: Camera suitable for full pipeline")

    return cap, actual_fps


def stream_frames(cap, actual_fps):
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Convert BGR to RGB immediately
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display feed with fps overlay (still in BGR for display)
        display = frame.copy()
        cv2.putText(display, f"FPS: {actual_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Camera Feed", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Run
cap, actual_fps = initialize_camera()
if cap:
    stream_frames(cap, actual_fps)