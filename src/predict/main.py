"""
Traffic Jam Detection System using YOLO Object Detection

This script analyzes traffic videos to detect:
- Vehicles (cars, vans, trucks, buses)
- Traffic state (FREE_FLOW, CONGESTED, TRAFFIC_JAM)
- Speed and spacing metrics

Key concepts:
1. Traffic zone: Only analyze bottom portion (cars closer to camera)
2. Spacing: Distance between vehicles (tight spacing = potential jam)
3. Speed: Movement between frames (slow/stopped = jam)
"""

from ultralytics import YOLO  # YOLO model for object detection
from pathlib import Path      # Handle file paths
import cv2                     # OpenCV for video processing
import numpy as np             # Numerical operations
from collections import Counter # Count traffic states

# ==================== HELPER FUNCTIONS ====================

def is_in_traffic_zone(box, img_height, zone_start_ratio=0.67):
    """
    Check if vehicle is in traffic analysis zone (bottom portion of frame)

    Why? Perspective makes distant cars (top of frame) look closer together.
    Only analyzing bottom portion gives accurate spacing measurements.

    Args:
        box: Bounding box [x1, y1, x2, y2] where:
             - x1, y1 = top-left corner
             - x2, y2 = bottom-right corner
        img_height: Total height of video frame
        zone_start_ratio: Where zone begins (0.67 = starts at 67% from top)

    Returns:
        True if vehicle center is in traffic zone, False otherwise
    """
    # Calculate center Y coordinate of bounding box
    # box[1] = y1 (top), box[3] = y2 (bottom)
    box_center_y = (box[1] + box[3]) / 2

    # Calculate zone threshold line
    # Example: 1080px height * 0.67 = 723px (zone starts at line 723)
    zone_threshold = img_height * zone_start_ratio

    # Vehicle is in zone if its center is below threshold line
    return box_center_y >= zone_threshold

def calculate_avg_spacing(boxes, img_height):
    """
    Calculate average vertical spacing between vehicles

    This measures how far apart cars are (in pixels). Tight spacing indicates
    congestion or traffic jam.

    Args:
        boxes: List of bounding boxes [[x1, y1, x2, y2], ...]
        img_height: Height of frame (used for normalization later)

    Returns:
        Average gap in pixels, or None if < 2 vehicles
    """
    # Need at least 2 cars to measure spacing
    if len(boxes) < 2:
        return None

    # Extract center Y coordinate of each car
    # Example: If box = [100, 200, 150, 250]
    #          center_y = (200 + 250) / 2 = 225
    # Sort top to bottom (smallest y to largest y)
    y_centers = sorted([((b[1] + b[3]) / 2) for b in boxes])

    # Calculate gap between each consecutive pair
    # Example: cars at y=[100, 180, 220]
    #          gaps = [180-100, 220-180] = [80, 40]
    # This represents the space between bumpers (simplified)
    gaps = [y_centers[i+1] - y_centers[i] for i in range(len(y_centers)-1)]

    # Return average gap
    # Example: [80, 40] → mean = 60 pixels
    return np.mean(gaps) if gaps else None

def calculate_avg_speed(boxes, prev_boxes, fps=30):
    """
    Calculate average speed of vehicles by tracking movement between frames

    Speed = how far vehicles move between consecutive frames.
    Slow speed = traffic jam, High speed = free flowing traffic.

    How it works:
    1. For each car in current frame, find its position in previous frame
    2. Calculate distance moved (displacement)
    3. Average all displacements = overall speed

    Args:
        boxes: Current frame vehicle boxes [[x1, y1, x2, y2], ...]
        prev_boxes: Previous frame vehicle boxes
        fps: Frames per second (to convert per-frame to per-second)

    Returns:
        Average speed in pixels/second, or None if insufficient data
    """
    # Need previous frame and at least 2 vehicles to calculate speed
    if not prev_boxes or len(boxes) < 2:
        return None

    # Store how far each vehicle moved
    displacements = []

    # Process each vehicle in current frame
    for box in boxes:
        # Calculate center point of current vehicle
        # box = [x1, y1, x2, y2]
        # center = ((x1+x2)/2, (y1+y2)/2)
        box_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

        # Find which vehicle in previous frame is the SAME vehicle
        # We do this by finding the closest position (simple tracking)
        min_dist = float('inf')  # Start with infinity (very large number)

        for prev_box in prev_boxes:
            # Calculate center of previous frame vehicle
            prev_center = ((prev_box[0] + prev_box[2]) / 2, (prev_box[1] + prev_box[3]) / 2)

            # Calculate Euclidean distance between centers
            # dist = sqrt((x2-x1)² + (y2-y1)²) - Pythagorean theorem
            # Example: current=(100,200), prev=(105,210)
            #          dist = sqrt((100-105)² + (200-210)²) = sqrt(25+100) = 11.2 pixels
            dist = np.sqrt((box_center[0] - prev_center[0])**2 + (box_center[1] - prev_center[1])**2)

            # Keep track of minimum distance (closest match)
            # The closest previous box is likely the same vehicle
            if dist < min_dist:
                min_dist = dist  # This is how far the vehicle moved

        # Save this vehicle's displacement
        displacements.append(min_dist)

    # Calculate average displacement across all vehicles
    # Example: displacements = [5, 8, 6] → mean = 6.3 pixels/frame
    avg_displacement = np.mean(displacements) if displacements else 0

    # Convert from pixels/frame to pixels/second
    # Example: 6.3 pixels/frame * 30 fps = 189 pixels/second
    # Higher number = faster traffic, Lower = slower/jammed
    return avg_displacement * fps

def classify_traffic(vehicle_count, avg_spacing, avg_speed, img_height):
    """
    Classify traffic state: FREE_FLOW, CONGESTED, or TRAFFIC_JAM

    Uses two main indicators:
    1. Speed: How fast vehicles are moving
    2. Spacing: How close vehicles are together

    Logic: If EITHER speed is slow OR spacing is tight → classify as jam/congested
          (Using OR instead of AND makes detection more sensitive)

    Args:
        vehicle_count: Number of vehicles in traffic zone
        avg_spacing: Average gap between vehicles (pixels)
        avg_speed: Average movement speed (pixels/second)
        img_height: Frame height for normalizing spacing

    Returns:
        (state_name, color_BGR) tuple
    """
    # ========== TUNING PARAMETERS ==========
    # Adjust these to change sensitivity

    VEHICLE_THRESHOLD = 5      # Min vehicles to analyze (< 5 cars = assume free flow)
    JAM_SPEED = 3.0           # Max speed for jam (px/s). INCREASE to detect more jams
    JAM_SPACING = 0.08        # Max spacing ratio for jam (0.08 = 8% of frame). INCREASE to detect more jams
    CONGESTED_SPEED = 12.0    # Max speed for congestion
    CONGESTED_SPACING = 0.15  # Max spacing ratio for congestion (15% of frame)

    # ========== CLASSIFICATION LOGIC ==========

    # Rule 1: Too few vehicles = free flowing road
    if vehicle_count < VEHICLE_THRESHOLD:
        return "FREE_FLOW", (0, 255, 0)  # Green color in BGR

    # Normalize spacing to percentage of frame height
    # Example: 60px gap on 1080px frame = 60/1080 = 0.055 (5.5%)
    # This makes thresholds work regardless of video resolution
    spacing_ratio = (avg_spacing / img_height) if avg_spacing else 1.0

    # Rule 2 & 3: Check speed AND spacing
    if avg_speed is not None:
        # TRAFFIC JAM: Very slow OR very tight spacing
        # Example: speed=2.5 (< 3.0) OR spacing=0.07 (< 0.08) → JAM
        if avg_speed < JAM_SPEED or spacing_ratio < JAM_SPACING:
            return "TRAFFIC_JAM", (0, 0, 255)  # Red

        # CONGESTED: Moderate slow OR moderate tight
        # Example: speed=10 (< 12.0) OR spacing=0.12 (< 0.15) → CONGESTED
        elif avg_speed < CONGESTED_SPEED or spacing_ratio < CONGESTED_SPACING:
            return "CONGESTED", (0, 165, 255)  # Orange
    else:
        # No speed data available, use spacing only
        if spacing_ratio < JAM_SPACING:
            return "TRAFFIC_JAM", (0, 0, 255)
        elif spacing_ratio < CONGESTED_SPACING:
            return "CONGESTED", (0, 165, 255)

    # Rule 4: Default to free flow
    return "FREE_FLOW", (0, 255, 0)  # Green

# ==================== MAIN EXECUTION ====================

# Step 1: Load trained YOLO model
# This model was trained on VisDrone dataset to detect vehicles
model = YOLO("runs/detect/train/weights/best.pt")

# Step 2: Set input video path
video_path = "src/dataset/traffic_jakarta_footage_2.mp4"

# Step 3: Create output directory
output_dir = Path("src/predict/VisDrone")
output_dir.mkdir(parents=True, exist_ok=True)  # Create if doesn't exist

print(f"Running prediction on: {video_path}")
print(f"Output directory: {output_dir}")

# Step 4: Run YOLO prediction
# stream=True: Process frame-by-frame (memory efficient for videos)
# conf=0.25: Only keep detections with ≥25% confidence
# iou=0.45: Non-Maximum Suppression threshold (removes duplicate boxes)
# verbose=False: Don't print progress for each frame
results_generator = model.predict(
    source=video_path,
    stream=True,       # Generator - yields one frame at a time
    conf=0.25,         # Confidence threshold
    iou=0.45,          # NMS IOU threshold
    verbose=False
)

# Step 5: Get video properties for output
# Open video temporarily to read metadata
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))          # Frames per second (e.g., 30)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # Width in pixels (e.g., 1920)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Height in pixels (e.g., 1080)
cap.release()  # Close video (don't need it open yet)

# Step 6: Prepare output video writer
new_video_name = Path(video_path).stem + "_annotated.mp4"  # Add "_annotated" suffix
output_video_path = output_dir / "results" / new_video_name
output_video_path.parent.mkdir(parents=True, exist_ok=True)

# H.264 codec (avc1) - widely compatible, especially on Mac
fourcc = cv2.VideoWriter_fourcc(*'avc1')
# Create video writer: (path, codec, fps, dimensions)
out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

# Step 7: Define which object classes to track
# YOLO can detect many objects, we only care about vehicles
vehicle_classes = ['car', 'van', 'truck', 'bus']

# Step 8: Initialize tracking variables
prev_boxes = []         # Store previous frame boxes for speed calculation
frame_idx = 0          # Current frame number
traffic_states = []    # Store state of each frame for statistics

print("Processing frames and analyzing traffic...")

# Step 9: Main processing loop - analyze each frame
# results_generator yields one frame at a time
for result in results_generator:
    # Get original frame image (before YOLO drew boxes on it)
    frame = result.orig_img.copy()

    # ========== STEP 9A: Extract vehicle detections ==========
    # YOLO detected many objects (people, cars, bikes, etc.)
    # We only want vehicles for traffic analysis
    all_vehicle_boxes = []
    for box in result.boxes:
        # Get class name (e.g., "car", "person", "bicycle")
        class_name = result.names[int(box.cls)]

        # Only keep vehicle classes
        if class_name in vehicle_classes:
            # box.xyxy = [x1, y1, x2, y2] coordinates
            # .cpu().numpy() converts from GPU tensor to numpy array
            all_vehicle_boxes.append(box.xyxy[0].cpu().numpy())

    # ========== STEP 9B: Filter to traffic zone ==========
    # Why? Distant cars (top of frame) look close due to perspective
    # Only analyze bottom portion where measurements are accurate

    # LOWER value = BIGGER zone (starts higher in frame)
    zone_start_ratio = 0.33  # Bottom 2/3 of frame (CURRENT)
    # Examples:
    # 0.67 = bottom 1/3 (smaller zone, only very close cars)
    # 0.5  = bottom 1/2 (medium zone)
    # 0.33 = bottom 2/3 (bigger zone, includes more cars)
    # 0.25 = bottom 3/4 (largest zone, includes distant cars)

    # Filter: keep only boxes where center is in traffic zone
    traffic_zone_boxes = [box for box in all_vehicle_boxes if is_in_traffic_zone(box, height, zone_start_ratio)]

    # ========== STEP 9C: Calculate traffic metrics ==========
    vehicle_count = len(traffic_zone_boxes)  # How many cars in zone

    # Spacing: average gap between vehicles (pixels)
    avg_spacing = calculate_avg_spacing(traffic_zone_boxes, height)

    # Speed: how far vehicles moved since last frame (pixels/second)
    # Skip first frame (no previous frame to compare)
    avg_speed = calculate_avg_speed(traffic_zone_boxes, prev_boxes, fps) if frame_idx > 0 else None

    # ========== STEP 9D: Classify traffic state ==========
    # Based on speed + spacing, determine: FREE_FLOW, CONGESTED, or TRAFFIC_JAM
    traffic_state, color = classify_traffic(vehicle_count, avg_spacing, avg_speed, height)
    traffic_states.append(traffic_state)  # Save for statistics

    # ========== STEP 9E: Visualize results on frame ==========

    # Start with YOLO's annotated frame (with bounding boxes already drawn)
    annotated_frame = result.plot()

    # Draw yellow line showing traffic zone boundary
    zone_y = int(height * zone_start_ratio)  # Calculate Y coordinate of zone line
    cv2.line(annotated_frame, (0, zone_y), (width, zone_y), (255, 255, 0), 2)  # Yellow horizontal line
    cv2.putText(annotated_frame, "TRAFFIC ZONE", (width - 200, zone_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Draw info overlay box (black background for text readability)
    overlay_height = 130
    # Rectangle: top-left (10,10), bottom-right (500, 130), black, filled
    cv2.rectangle(annotated_frame, (10, 10), (500, overlay_height), (0, 0, 0), -1)

    # Line 1: Traffic state (in state color: red/orange/green)
    cv2.putText(annotated_frame, f"Traffic State: {traffic_state}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Line 2: Vehicle count (white text)
    cv2.putText(annotated_frame, f"Vehicles: {vehicle_count}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Line 3: Metrics for tuning (speed and spacing)
    # These help you adjust thresholds in classify_traffic()
    spacing_ratio = (avg_spacing / height) if avg_spacing else 0
    speed_display = f"{avg_speed:.1f}" if avg_speed else "N/A"
    cv2.putText(annotated_frame, f"Speed: {speed_display} px/s | Spacing: {spacing_ratio:.3f}", (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Line 4: Frame number
    cv2.putText(annotated_frame, f"Frame: {frame_idx}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # ========== STEP 9F: Save frame to output video ==========
    out.write(annotated_frame)

    # ========== STEP 9G: Prepare for next frame ==========
    # Save current boxes for speed calculation in next iteration
    prev_boxes = traffic_zone_boxes

    # Increment frame counter
    frame_idx += 1

    # Print progress every 50 frames
    if frame_idx % 50 == 0:
        print(f"Processed {frame_idx} frames...")

# Step 10: Close video writer
out.release()  # Finalize and save the output video

# ========== STEP 11: Calculate and display statistics ==========

# Count how many frames had each traffic state
# Example: {'FREE_FLOW': 450, 'CONGESTED': 100, 'TRAFFIC_JAM': 58}
from collections import Counter
state_counts = Counter(traffic_states)
total_frames = len(traffic_states)

# Display summary report
print(f"\n{'='*50}")
print(f"Analysis Complete!")
print(f"{'='*50}")
print(f"Total frames: {total_frames}")
print(f"Traffic states distribution:")

# Show each state with count and percentage
# most_common() sorts by frequency (most common first)
for state, count in state_counts.most_common():
    percentage = (count / total_frames) * 100
    print(f"  {state}: {count} frames ({percentage:.1f}%)")

print(f"\nResults saved to: {output_video_path}")

# ========== END OF ANALYSIS ==========
# The output video now contains:
# - Bounding boxes around detected vehicles
# - Traffic zone boundary (yellow line)
# - Real-time traffic state classification
# - Speed and spacing metrics for each frame
