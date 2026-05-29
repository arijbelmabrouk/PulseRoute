import cv2

cap = cv2.VideoCapture(0)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

# Some cameras report 0 fps — measure it manually if that happens
if fps == 0:
    fps = 30  # safe fallback, we'll measure real fps during capture

print(f"Camera detected: {width}x{height} @ {fps}fps")

# Classify what we're working with
if fps < 15:
    print("WARNING: FPS too low for reliable heart rate detection")
elif fps < 30:
    print("ACCEPTABLE: Signal quality may be reduced")
else:
    print("GOOD: Camera suitable for full pipeline")