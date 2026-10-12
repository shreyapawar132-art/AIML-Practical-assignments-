# Assignment 9: Object / Motion Detection from CCTV Footage
# This version creates a small CCTV-style input video automatically if one does not exist.
import cv2
import numpy as np
from pathlib import Path

INPUT = "cctv_input.mp4"
OUTPUT = "cctv_annotated.mp4"

def create_demo_video(path, frames=100, width=640, height=360):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 20.0, (width, height))

    for i in range(frames):
        frame = np.full((height, width, 3), 35, dtype=np.uint8)

        # CCTV-style background
        cv2.rectangle(frame, (0, 250), (width, height), (55, 55, 55), -1)
        cv2.line(frame, (0, 250), (width, 250), (90, 90, 90), 2)

        # Moving object 1
        x1 = 30 + (i * 4) % 500
        cv2.rectangle(frame, (x1, 150, x1 + 70, 215), (180, 180, 180), -1)

        # Moving object 2
        x2 = 520 - (i * 3) % 400
        cv2.rectangle(frame, (x2, 80, x2 + 55, 135), (120, 120, 120), -1)

        cv2.putText(frame, "CCTV CAMERA 01", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)

        writer.write(frame)

    writer.release()

if not Path(INPUT).exists():
    print("Input video not found. Creating demo CCTV video...")
    create_demo_video(INPUT)

cap = cv2.VideoCapture(INPUT)

if not cap.isOpened():
    raise RuntimeError("Could not open CCTV input video.")

back_sub = cv2.createBackgroundSubtractorMOG2(
    history=40, varThreshold=40, detectShadows=False
)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(OUTPUT, fourcc, fps, (width, height))

frames = 0
max_regions = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frames += 1
    mask = back_sub.apply(frame)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    regions = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 250:
            regions += 1
            cv2.rectangle(
                frame, (x, y), (x + w, y + h), (0, 80, 220), 2
            )

    max_regions = max(max_regions, regions)
    cv2.putText(
        frame, f"Moving Regions: {regions}", (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    writer.write(frame)

    cv2.imshow("CCTV Detection", frame)
    if cv2.waitKey(20) & 0xFF == 27:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

print("Frames processed:", frames)
print("Maximum detected moving regions in one frame:", max_regions)
print("Annotated video:", OUTPUT)
