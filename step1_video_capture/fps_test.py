import cv2
import time
from collections import deque

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)

timestamps = deque(maxlen=30)
fps = 0.0  # default before deque fills up

while True:
    ret, frame = cap.read()
    if not ret:
        break

    timestamps.append(time.time())
    if len(timestamps) >= 2:
        fps = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("FPS Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()