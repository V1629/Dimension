import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
import time

# ===================================================================
#  3D CUBOID HAND TRACKER
#  - Uses multiple fingers from both hands (index, middle, pinky tips + wrist)
#  - Draws a 3D cuboid wireframe between the two hands
#  - Rotation: twist thumb+index to rotate the cuboid
#  - Smooth tracking with exponential moving average (EMA)
#  - Multiple visual filters inside the shape
# ===================================================================

# ===== State =====
latest_result = None
current_filter = 0
filter_names = ["RAINBOW", "SKETCH", "PIXELATE", "EDGE-GLOW", "NONE"]
rainbow_offset = 0

# Smoothing buffers (EMA)
smooth_points = {}  # key -> (x, y) smoothed
SMOOTH_FACTOR = 0.25  # lower = smoother but more lag (was 0.35)

# Mode state with debouncing
current_mode = "NONE"       # "NONE", "2D", "3D"
candidate_mode = "NONE"
mode_frame_counter = 0
MODE_DEBOUNCE_FRAMES = 5    # require N stable frames before switching

# Per-finger extension state smoothing
finger_state_counters = {}  # key -> int (positive = extending, negative = folding)
FINGER_DEBOUNCE = 3         # frames required to toggle finger state
finger_states = {}          # key -> bool (smoothed extended state)

# ===== Smoothing helper =====
def smooth(key, new_point):
    """Exponential moving average for smooth landmark tracking."""
    if key not in smooth_points:
        smooth_points[key] = new_point
        return new_point
    old = smooth_points[key]
    sx = int(old[0] * (1 - SMOOTH_FACTOR) + new_point[0] * SMOOTH_FACTOR)
    sy = int(old[1] * (1 - SMOOTH_FACTOR) + new_point[1] * SMOOTH_FACTOR)
    smooth_points[key] = (sx, sy)
    return (sx, sy)

smooth_floats = {}  # key -> float smoothed
def smooth_float(key, new_val, factor=None):
    """EMA smoothing for scalar values (rotation, depth)."""
    f = factor if factor is not None else SMOOTH_FACTOR
    if key not in smooth_floats:
        smooth_floats[key] = new_val
        return new_val
    old = smooth_floats[key]
    smoothed = old * (1 - f) + new_val * f
    smooth_floats[key] = smoothed
    return smoothed

def debounce_finger(hand_id, finger_idx, raw_extended):
    """Debounce finger extension state to avoid noise."""
    key = f"h{hand_id}_f{finger_idx}"
    if key not in finger_state_counters:
        finger_state_counters[key] = 0
        finger_states[key] = False

    if raw_extended:
        finger_state_counters[key] = min(finger_state_counters[key] + 1, FINGER_DEBOUNCE + 1)
    else:
        finger_state_counters[key] = max(finger_state_counters[key] - 1, -(FINGER_DEBOUNCE + 1))

    if finger_state_counters[key] >= FINGER_DEBOUNCE:
        finger_states[key] = True
    elif finger_state_counters[key] <= -FINGER_DEBOUNCE:
        finger_states[key] = False

    return finger_states[key]

# ===== Helpers =====
def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def get_px(landmarks, idx, w, h):
    lm = landmarks[idx]
    return int(lm.x * w), int(lm.y * h)

def get_z(landmarks, idx):
    """Get relative Z depth of a landmark (closer = more negative)."""
    return landmarks[idx].z

def angle_between(p1, p2):
    """Angle in degrees from p1 to p2."""
    return math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))

def rotate_point(point, center, angle_deg):
    """Rotate a point around a center by angle_deg degrees."""
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    rx = int(cos_a * dx - sin_a * dy + center[0])
    ry = int(sin_a * dx + cos_a * dy + center[1])
    return (rx, ry)

# ===== Finger extension detection =====
# Landmark indices per finger:
#   Thumb:  1(CMC), 2(MCP), 3(IP),  4(TIP)
#   Index:  5(MCP), 6(PIP), 7(DIP), 8(TIP)
#   Middle: 9(MCP), 10(PIP),11(DIP),12(TIP)
#   Ring:  13(MCP),14(PIP),15(DIP),16(TIP)
#   Pinky: 17(MCP),18(PIP),19(DIP),20(TIP)

FINGER_TIP_IDS  = [4, 8, 12, 16, 20]
FINGER_PIP_IDS  = [3, 6, 10, 14, 18]  # IP for thumb, PIP for others
FINGER_NAMES    = ["thumb", "index", "middle", "ring", "pinky"]

# Which landmarks belong to each finger (for selective rendering)
FINGER_LANDMARKS = {
    0: [1, 2, 3, 4],      # thumb
    1: [5, 6, 7, 8],      # index
    2: [9, 10, 11, 12],   # middle
    3: [13, 14, 15, 16],  # ring
    4: [17, 18, 19, 20],  # pinky
}

def detect_extended_fingers(hand_landmarks, hand_id):
    """
    Detect which fingers are extended using landmark positions.
    Returns: list of 5 bools [thumb, index, middle, ring, pinky]
    Uses debouncing to avoid noise-triggered toggling.
    """
    extended_raw = [False] * 5

    # Thumb: compare distance of tip(4) vs IP(3) from wrist(0)
    wrist = hand_landmarks[0]
    thumb_tip = hand_landmarks[4]
    thumb_ip = hand_landmarks[3]
    dist_tip = math.sqrt((thumb_tip.x - wrist.x)**2 + (thumb_tip.y - wrist.y)**2)
    dist_ip = math.sqrt((thumb_ip.x - wrist.x)**2 + (thumb_ip.y - wrist.y)**2)
    extended_raw[0] = dist_tip > dist_ip * 1.15  # 15% margin

    # Index, Middle, Ring, Pinky: tip.y < pip.y means pointing up
    for i in range(1, 5):
        tip = hand_landmarks[FINGER_TIP_IDS[i]]
        pip = hand_landmarks[FINGER_PIP_IDS[i]]
        extended_raw[i] = tip.y < pip.y - 0.02  # small threshold for noise

    # Apply per-finger debouncing
    extended_debounced = []
    for i in range(5):
        extended_debounced.append(debounce_finger(hand_id, i, extended_raw[i]))

    return extended_debounced

# ===== Draw hand skeleton (only extended fingers) =====
def draw_hand_skeleton(frame, hand_landmarks, w, h, hand_id, extended_fingers):
    """Draw landmarks only for extended fingers. Wrist + palm base always shown."""
    # Build set of landmarks that should be visible
    visible_landmarks = {0}  # wrist always visible

    for i in range(5):
        if extended_fingers[i]:
            for lm_id in FINGER_LANDMARKS[i]:
                visible_landmarks.add(lm_id)
            # Also add MCP for structural connection to wrist
            if i > 0:
                visible_landmarks.add(FINGER_LANDMARKS[i][0])

    # All possible connections
    ALL_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17)
    ]

    color_glow = (0, 180, 0) if hand_id == 0 else (180, 100, 0)
    color_line = (0, 255, 100) if hand_id == 0 else (255, 200, 50)

    # Draw only connections where BOTH endpoints are visible
    for c in ALL_CONNECTIONS:
        if c[0] in visible_landmarks and c[1] in visible_landmarks:
            s = hand_landmarks[c[0]]
            e = hand_landmarks[c[1]]
            x1, y1 = int(s.x*w), int(s.y*h)
            x2, y2 = int(e.x*w), int(e.y*h)
            cv2.line(frame, (x1,y1), (x2,y2), color_glow, 4, cv2.LINE_AA)
            cv2.line(frame, (x1,y1), (x2,y2), color_line, 2, cv2.LINE_AA)

    # Draw only visible landmark dots
    for i in visible_landmarks:
        lm = hand_landmarks[i]
        cx, cy = int(lm.x*w), int(lm.y*h)
        if i in FINGER_TIP_IDS:
            cv2.circle(frame, (cx,cy), 7, (0,255,255), cv2.FILLED, cv2.LINE_AA)
            cv2.circle(frame, (cx,cy), 9, (0,200,200), 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, (cx,cy), 4, (0,220,180), cv2.FILLED, cv2.LINE_AA)

# ===== Get key points from a hand using extended fingers =====
def get_hand_points(landmarks, w, h, hand_id, extended_fingers):
    """
    Returns key points from a hand based on which fingers are extended.
    Always includes wrist and thumb/index for rotation.
    'top_point' is the highest (lowest y) extended fingertip.
    """
    wrist = smooth(f"h{hand_id}_wrist", get_px(landmarks, 0, w, h))
    thumb_tip = smooth(f"h{hand_id}_thumb", get_px(landmarks, 4, w, h))
    index_tip = smooth(f"h{hand_id}_index", get_px(landmarks, 8, w, h))

    # Collect all extended fingertip positions
    extended_tips = []
    tip_indices = [4, 8, 12, 16, 20]
    for i in range(5):
        if extended_fingers[i]:
            tip = smooth(f"h{hand_id}_tip{i}", get_px(landmarks, tip_indices[i], w, h))
            extended_tips.append(tip)

    # Top point = highest (min y) extended fingertip, or index if none
    if extended_tips:
        top_point = min(extended_tips, key=lambda p: p[1])
    else:
        top_point = index_tip

    # Average Z for depth estimation (use wrist + extended tips)
    z_vals = [get_z(landmarks, 0)]
    for i in range(5):
        if extended_fingers[i]:
            z_vals.append(get_z(landmarks, tip_indices[i]))
    avg_z = sum(z_vals) / len(z_vals)

    return {
        "top": top_point,
        "index": index_tip,
        "wrist": wrist,
        "thumb": thumb_tip,
        "extended_tips": extended_tips,
        "z": avg_z
    }

# ===== Draw 3D cuboid wireframe =====
def draw_cuboid(frame, front_pts, back_pts, color=(0, 255, 255), thickness=2):
    """
    Draw a cuboid wireframe given 4 front face points and 4 back face points.
    front_pts & back_pts are lists of 4 (x,y) tuples each.
    """
    # Draw front face
    for i in range(4):
        cv2.line(frame, front_pts[i], front_pts[(i+1)%4], color, thickness, cv2.LINE_AA)
    # Draw back face (slightly dimmer)
    back_color = tuple(max(0, c-60) for c in color)
    for i in range(4):
        cv2.line(frame, back_pts[i], back_pts[(i+1)%4], back_color, thickness, cv2.LINE_AA)
    # Draw connecting edges (depth lines)
    depth_color = tuple(max(0, c-30) for c in color)
    for i in range(4):
        cv2.line(frame, front_pts[i], back_pts[i], depth_color, thickness-1 if thickness > 1 else 1, cv2.LINE_AA)

# ===== Build cuboid from two hands (3D mode) =====
def build_cuboid_from_hands(hand1, hand2, rotation_angle):
    """
    Build front and back face of cuboid from two hand point sets.
    Uses the highest extended fingertip as top corners and wrists as bottom.
    """
    # Determine left/right hand by X position
    if hand1["top"][0] < hand2["top"][0]:
        left, right = hand1, hand2
    else:
        left, right = hand2, hand1

    # Front face: top corners from highest extended tips, bottom from wrists
    front = [
        left["top"],
        right["top"],
        right["wrist"],
        left["wrist"]
    ]

    # Calculate center for rotation
    cx = sum(p[0] for p in front) // 4
    cy = sum(p[1] for p in front) // 4
    center = (cx, cy)

    # Apply rotation from thumb-index angle
    if abs(rotation_angle) > 2:  # Dead zone to avoid jitter
        front = [rotate_point(p, center, rotation_angle * 0.3) for p in front]

    # Depth offset for back face (smooth Z-based depth)
    z_diff = abs(left["z"] - right["z"])
    depth_offset = int(smooth_float("depth", max(20, min(80, z_diff * 800)), 0.2))

    # Back face shifted diagonally (perspective effect)
    dx = int(depth_offset * 0.7)
    dy = int(depth_offset * 0.5)
    back = [(p[0]+dx, p[1]-dy) for p in front]

    return front, back, center

# ===== Build 2D rectangle from two hands (2D mode) =====
def build_2d_rect_from_hands(hand1, hand2, rotation_angle):
    """
    Build a smooth 2D rectangle from thumb+index of both hands.
    Uses index tips as top corners and wrists as bottom corners.
    """
    # Determine left/right hand by X position
    if hand1["index"][0] < hand2["index"][0]:
        left, right = hand1, hand2
    else:
        left, right = hand2, hand1

    # 2D rectangle: index tips as top, wrists as bottom
    corners = [
        left["index"],
        right["index"],
        right["wrist"],
        left["wrist"]
    ]

    # Calculate center for rotation
    cx = sum(p[0] for p in corners) // 4
    cy = sum(p[1] for p in corners) // 4
    center = (cx, cy)

    # Apply rotation from thumb-index angle (gentler in 2D mode)
    if abs(rotation_angle) > 2:
        corners = [rotate_point(p, center, rotation_angle * 0.2) for p in corners]

    return corners, center

# ===== Filter functions =====
def create_rainbow_fill(width, height, offset):
    if width <= 0 or height <= 0:
        return None
    fill = np.zeros((height, width, 3), dtype=np.uint8)
    stripe_w = 18
    colors = [
        (0,0,255), (0,127,255), (0,255,255),
        (0,255,0), (255,255,0), (255,0,0), (255,0,127)
    ]
    for y in range(height):
        for x in range(width):
            idx = ((x + y + offset) // stripe_w) % len(colors)
            fill[y, x] = colors[idx]
    return fill

def apply_filter_in_polygon(frame, pts, filter_idx):
    """Apply filter inside a polygon defined by pts."""
    global rainbow_offset
    pts_arr = np.array(pts, dtype=np.int32)
    x, y, rw, rh = cv2.boundingRect(pts_arr)
    h, w = frame.shape[:2]
    x, y = max(0, x), max(0, y)
    rw = min(rw, w - x)
    rh = min(rh, h - y)
    if rw < 10 or rh < 10:
        return

    # Create mask from polygon
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts_arr], 255)

    if filter_idx == 0:  # RAINBOW
        rainbow_offset += 3
        fill = create_rainbow_fill(rw, rh, rainbow_offset)
        if fill is not None:
            roi = frame[y:y+rh, x:x+rw]
            fill_resized = fill[:roi.shape[0], :roi.shape[1]]
            roi_mask = mask[y:y+rh, x:x+rw]
            blended = cv2.addWeighted(roi, 0.3, fill_resized, 0.7, 0)
            roi[roi_mask > 0] = blended[roi_mask > 0]

    elif filter_idx == 1:  # SKETCH
        roi = frame[y:y+rh, x:x+rw]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv, (21,21), 0)
        sketch = cv2.divide(gray, 255-blur, scale=256)
        sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
        roi_mask = mask[y:y+rh, x:x+rw]
        roi[roi_mask > 0] = sketch_bgr[roi_mask > 0]

    elif filter_idx == 2:  # PIXELATE
        roi = frame[y:y+rh, x:x+rw]
        ph, pw = roi.shape[:2]
        pix_size = 10
        small = cv2.resize(roi, (max(1,pw//pix_size), max(1,ph//pix_size)), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (pw, ph), interpolation=cv2.INTER_NEAREST)
        roi_mask = mask[y:y+rh, x:x+rw]
        roi[roi_mask > 0] = pixelated[roi_mask > 0]

    elif filter_idx == 3:  # EDGE-GLOW
        roi = frame[y:y+rh, x:x+rw]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_colored = np.zeros_like(roi)
        edge_colored[edges > 0] = (255, 255, 0)
        glow = cv2.GaussianBlur(edge_colored, (7,7), 0)
        result = cv2.addWeighted(edge_colored, 1.0, glow, 0.8, 0)
        roi_mask = mask[y:y+rh, x:x+rw]
        roi[roi_mask > 0] = result[roi_mask > 0]

# ===== MediaPipe callback =====
def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

# ===== Setup MediaPipe =====
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=result_callback
)
detector = vision.HandLandmarker.create_from_options(options)

# ===== Webcam =====
cap = cv2.VideoCapture(0)
timestamp = 0
prev_time = time.time()

print("=" * 60)
print("  HAND SHAPE TRACKER (2D / 3D)")
print("=" * 60)
print("  Show BOTH hands to generate shapes between them")
print("  Thumb + Index only  → smooth 2D rectangle")
print("  3+ fingers extended → 3D cuboid outline")
print("  Rotate thumb+index  → twist the shape")
print("")
print("  'f' = cycle filters  |  'x' = quit")
print("=" * 60)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp += 1
    detector.detect_async(mp_image, timestamp)

    # ===== Per-frame finger detection =====
    all_extended = []  # list of extended_fingers per hand
    frame_mode = "NONE"

    if latest_result and latest_result.hand_landmarks:
        hands = latest_result.hand_landmarks

        for i, hand_lm in enumerate(hands):
            ext = detect_extended_fingers(hand_lm, i)
            all_extended.append(ext)
            draw_hand_skeleton(frame, hand_lm, w, h, i, ext)

        if len(hands) >= 2 and len(all_extended) >= 2:
            # Count extended fingers per hand
            ext1_count = sum(all_extended[0])
            ext2_count = sum(all_extended[1])

            # Check if exactly thumb+index on each hand → 2D mode
            is_2d_gesture = True
            for ext in all_extended[:2]:
                if not (ext[0] and ext[1] and not ext[2] and not ext[3] and not ext[4]):
                    is_2d_gesture = False
                    break

            if is_2d_gesture:
                frame_mode = "2D"
            elif ext1_count >= 3 or ext2_count >= 3:
                frame_mode = "3D"

    # ===== Mode debouncing =====
    if frame_mode == candidate_mode:
        mode_frame_counter += 1
    else:
        candidate_mode = frame_mode
        mode_frame_counter = 1

    if mode_frame_counter >= MODE_DEBOUNCE_FRAMES:
        current_mode = candidate_mode

    # ===== Render based on current stable mode =====
    if current_mode in ("2D", "3D") and latest_result and latest_result.hand_landmarks:
        hands = latest_result.hand_landmarks
        if len(hands) >= 2 and len(all_extended) >= 2:
            hp1 = get_hand_points(hands[0], w, h, 0, all_extended[0])
            hp2 = get_hand_points(hands[1], w, h, 1, all_extended[1])

            # Calculate rotation from thumb-index angle
            angle1 = angle_between(hp1["thumb"], hp1["index"])
            angle2 = angle_between(hp2["thumb"], hp2["index"])
            avg_angle = (angle1 + angle2) / 2.0
            rotation = smooth_float("rotation", avg_angle, 0.3)

            if current_mode == "2D":
                corners, center = build_2d_rect_from_hands(hp1, hp2, rotation)

                # Apply filter inside 2D rectangle
                if current_filter < 4:
                    apply_filter_in_polygon(frame, corners, current_filter)

                # Draw smooth 2D rectangle
                pts = np.array(corners, dtype=np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)
                for pt in corners:
                    cv2.circle(frame, pt, 5, (255, 255, 255), cv2.FILLED, cv2.LINE_AA)

            elif current_mode == "3D":
                front, back, center = build_cuboid_from_hands(hp1, hp2, rotation)

                # Apply filter inside front face
                if current_filter < 4:
                    apply_filter_in_polygon(frame, front, current_filter)

                # Draw 3D cuboid wireframe
                draw_cuboid(frame, front, back, (0, 255, 255), 2)
                for pt in front:
                    cv2.circle(frame, pt, 5, (255, 255, 255), cv2.FILLED, cv2.LINE_AA)
                for pt in back:
                    cv2.circle(frame, pt, 3, (180, 180, 180), cv2.FILLED, cv2.LINE_AA)

    # ===== HUD =====
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), (0, 0, 0), cv2.FILLED)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Mode display
    mode_colors = {"NONE": (120,120,120), "2D": (0,255,255), "3D": (0,255,0)}
    mode_text = f"MODE: {current_mode}"
    if current_mode == "2D":
        mode_text += " RECTANGLE"
    elif current_mode == "3D":
        mode_text += " CUBOID"
    else:
        mode_text += " (show both hands)"
    cv2.putText(frame, mode_text, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                mode_colors.get(current_mode, (200,200,200)), 1, cv2.LINE_AA)

    # Filter display
    filter_text = f"FILTER: {filter_names[current_filter]}  |  'f' to change"
    cv2.putText(frame, filter_text, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1, cv2.LINE_AA)

    # Finger count per hand
    if all_extended:
        fingers_str = " | ".join(
            [f"H{i+1}: {sum(e)}F" for i, e in enumerate(all_extended)]
        )
    else:
        fingers_str = "No hands"
    cv2.putText(frame, fingers_str, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1, cv2.LINE_AA)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 0.001)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (w-90, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1, cv2.LINE_AA)

    cv2.imshow("Hand Shape Tracker", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('x'):
        break
    elif key == ord('f'):
        current_filter = (current_filter + 1) % len(filter_names)
        print(f"Filter: {filter_names[current_filter]}")

cap.release()
cv2.destroyAllWindows()
