"""
Volleyball Detection - Dual Output Prediction

Input:  Single video file
Output: 1. Individual annotated frames (images)
        2. Annotated video
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import math
from collections import deque

# ==================== CONFIGURATION ====================

# Model
model_path = "runs/detect/volleyball/weights/best.pt"

# Rally detection settings
# Idea: a rally = ball is MOVING, not just visible. A held or resting ball should not count.
RALLY_BALL_CLASS_ID = 0       # Class ID for ball in your model (check model.names)
RALLY_MIN_SPEED_PX = 4        # Min ball movement per frame (pixels) to count as "moving".
                              # Raise if a held/rolling ball triggers rally; lower for far cameras.
RALLY_START_WINDOW_SEC = 0.5  # How far back (seconds) to look when deciding a rally has started
RALLY_START_MIN_MOVING = 5    # Moving-ball frames needed inside the start window to begin rally.
                              # Must be <= frames in the window (0.5s at 30fps = 15 frames).
RALLY_END_IDLE_SEC = 1.5      # Rally ends after the ball has not moved for this many seconds

# Rally display settings
RALLY_TEXT_SCALE = 1.5        # Text size
RALLY_TEXT_THICKNESS = 3      # Text thickness
RALLY_TEXT_COLOR = (0, 255, 255)  # Text color (BGR: yellow)
RALLY_BG_COLOR = (0, 0, 0)    # Background color (BGR: black)
RALLY_MARGIN = 20             # Distance from top-right corner

# Input video
input_video = "src/dataset/Volleyball_3.mp4"

# Output directory
output_dir = Path("src/predict/volley/results")
frames_dir = output_dir / "frames"     # Individual frames
video_dir = output_dir / "video"       # Annotated video

# Create output directories
frames_dir.mkdir(parents=True, exist_ok=True)
video_dir.mkdir(parents=True, exist_ok=True)

# ==================== LOAD MODEL ====================

print("=" * 60)
print("VOLLEYBALL DETECTION - DUAL OUTPUT MODE")
print("=" * 60)
print(f"Model: {model_path}")
print(f"Input: {input_video}")
print(f"Output:")
print(f"  - Frames: {frames_dir}")
print(f"  - Video:  {video_dir}")
print("=" * 60)

model = YOLO(model_path)

# Check input exists
if not Path(input_video).exists():
    print(f"\n❌ ERROR: Video not found: {input_video}")
    exit(1)

# ==================== PROCESS VIDEO ====================

print("\nProcessing video...")

# Get video properties for output video writer
cap = cv2.VideoCapture(input_video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")

# Setup video writer for output video
output_video_name = Path(input_video).stem + "_annotated.mp4"
output_video_path = video_dir / output_video_name
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

# Run prediction with streaming
# stream=True: Process frame-by-frame (memory efficient for videos)
# conf=0.25  # Standard (good balance)
# conf=0.4   # Stricter (fewer false ball detections)
# conf=0.15  # Permissive (catch distant/occluded players)
# iou=0.45   # Standard NMS
# iou=0.3    # Aggressive (good for tight player groups)
# iou=0.6    # Permissive (allow overlapping detections)
# verbose=False: Don't print progress for each frame
results_generator = model.predict(
    source=input_video,
    imgsz=1024,
    conf=0.4,
    iou=0.3,
    stream=True,    # Stream mode - process frame by frame
    device="mps",
    verbose=False
)

# ==================== RALLY DETECTION STATE ====================
# Settings are in seconds so they work on any fps. Convert them to frame counts here.
# Example at 30fps: 0.5s -> 15 frames, 1.5s -> 45 frames.
# max(1, ...) guarantees at least 1 frame even if fps is very low.
window_frames_start = max(1, round(RALLY_START_WINDOW_SEC * fps))
end_idle_frames = max(1, round(RALLY_END_IDLE_SEC * fps))

# Sliding window of True/False ("was the ball moving?") for the most recent frames.
# deque(maxlen=N) automatically drops the oldest item when a new one is appended,
# so it always holds only the last `window_frames_start` results.
moving_history = deque(maxlen=window_frames_start)

# Memory between frames (None = "not seen yet"):
last_ball_center = None   # (x, y) pixel center of the ball the last time it was detected
last_ball_frame = None    # frame number when the ball was last detected
last_moving_frame = None  # frame number when the ball was last confirmed moving

# The current rally state. It only flips when the start or end rule below is met.
is_rally_active = False

# Process each frame
frame_idx = 0
total_detections = 0

print("\nProcessing frames:")
for result in results_generator:
    # Get annotated frame
    annotated_frame = result.plot()

    # ---------- RALLY STEP 1: is the ball moving in this frame? ----------
    # Default to "not moving". It only becomes True if the ball is found AND it moved enough.
    ball_moving = False

    # Keep only the ball detections (ignore players etc.).
    # result.boxes holds every detection; box.cls is its class id.
    ball_boxes = [b for b in result.boxes if int(b.cls[0]) == RALLY_BALL_CLASS_ID]

    if ball_boxes:
        # If the model found more than one "ball" (e.g. a spare ball or a false
        # detection), trust the one with the highest confidence score.
        best_ball = max(ball_boxes, key=lambda b: float(b.conf[0]))

        # xywh = [center_x, center_y, width, height]. We only need the center point.
        cx, cy = best_ball.xywh[0][:2].tolist()

        # We can only measure movement if we saw the ball before.
        # If it was missing for longer than end_idle_frames, the old position is
        # too old to compare, so we skip and just record the new position.
        if last_ball_center is not None and frame_idx - last_ball_frame <= end_idle_frames:
            # How many frames since the ball was last seen (1 = previous frame).
            gap = frame_idx - last_ball_frame

            # math.hypot(dx, dy) = straight-line distance = sqrt(dx² + dy²).
            # Dividing by gap gives pixels PER FRAME, so a ball hidden for a few
            # frames is not mistaken for a very fast ball.
            # Example: moved 12px over 3 frames -> 4 px/frame.
            speed = math.hypot(cx - last_ball_center[0], cy - last_ball_center[1]) / gap

            # Fast enough = in play. A ball held by a player or lying still stays below this.
            ball_moving = speed >= RALLY_MIN_SPEED_PX

        # Remember where and when we saw the ball, for the next frame's comparison.
        last_ball_center = (cx, cy)
        last_ball_frame = frame_idx

    # Record this frame's result in the sliding window (oldest one drops out automatically).
    moving_history.append(ball_moving)
    if ball_moving:
        last_moving_frame = frame_idx

    # ---------- RALLY STEP 2: update rally ON/OFF state ----------
    # This is "hysteresis": different rules for turning ON and turning OFF.
    # Using one shared threshold would make the text blink when values hover near it.
    if not is_rally_active:
        # START rule (quick but needs evidence):
        # sum() counts True values, e.g. [F, T, T, F, T] -> 3.
        # Turn on when enough recent frames had a moving ball.
        if sum(moving_history) >= RALLY_START_MIN_MOVING:
            is_rally_active = True
    elif frame_idx - last_moving_frame > end_idle_frames:
        # END rule (slow and forgiving):
        # Only turn off after the ball has not moved for RALLY_END_IDLE_SEC.
        # Short occlusions or missed detections mid-rally won't end it.
        # (last_moving_frame is never None here: the rally could only start after a moving frame.)
        is_rally_active = False

        # Forget old movement so the next rally must earn its own start.
        moving_history.clear()

    # ---------- RALLY STEP 3: draw "RALLY" text at top-right if active ----------
    if is_rally_active:
        text = "RALLY"
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Get text size for positioning
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, RALLY_TEXT_SCALE, RALLY_TEXT_THICKNESS)

        # Position at top-right with margin
        x = width - text_width - RALLY_MARGIN
        y = text_height + RALLY_MARGIN

        # Draw background rectangle
        cv2.rectangle(annotated_frame,
                     (x - 10, y - text_height - 10),
                     (x + text_width + 10, y + baseline + 10),
                     RALLY_BG_COLOR, -1)

        # Draw text
        cv2.putText(annotated_frame, text, (x, y), font, RALLY_TEXT_SCALE,
                   RALLY_TEXT_COLOR, RALLY_TEXT_THICKNESS, cv2.LINE_AA)

    # Output 1: Save as individual image
    frame_filename = f"frame_{frame_idx:05d}.jpg"
    frame_path = frames_dir / frame_filename
    cv2.imwrite(str(frame_path), annotated_frame)

    # Output 2: Write to video
    video_writer.write(annotated_frame)

    # Track statistics
    num_detections = len(result.boxes)
    total_detections += num_detections

    frame_idx += 1

    # Progress update
    if frame_idx % 50 == 0:
        progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
        print(f"  Frame {frame_idx}/{total_frames} ({progress:.1f}%) - {num_detections} detections")

# Close video writer
video_writer.release()

# ==================== SUMMARY ====================

print("\n" + "=" * 60)
print("PROCESSING COMPLETE!")
print("=" * 60)
print(f"Frames processed: {frame_idx}")
print(f"Total detections: {total_detections}")
if frame_idx > 0:
    print(f"Avg detections/frame: {total_detections/frame_idx:.2f}")

print(f"\n📁 Output 1 - Individual Frames:")
print(f"   Location: {frames_dir}")
print(f"   Files: {frame_idx} images (frame_00000.jpg to frame_{frame_idx-1:05d}.jpg)")

print(f"\n🎥 Output 2 - Annotated Video:")
print(f"   Location: {output_video_path}")
print(f"   Format: H.264 MP4 ({width}x{height} @ {fps}fps)")

print("\n" + "=" * 60)
