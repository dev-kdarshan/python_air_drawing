import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math

# =========================
# Kalman Filter (smooth tip)
# =========================
class KalmanFilter2D:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov = np.identity(4, np.float32) * 0.03
    def update(self, x, y):
        z = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(z)
        p = self.kf.predict()
        return int(p[0]), int(p[1])

kf = KalmanFilter2D()

# ==============
# Mediapipe Hands
# ==============
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# =========================
# Gesture + finger utilities
# =========================
def finger_states(hand_lm):
    pts = [(hand_lm.landmark[i].x, hand_lm.landmark[i].y) for i in range(21)]
    fingers = []
    fingers.append(pts[8][1] < pts[6][1])   # index
    fingers.append(pts[12][1] < pts[10][1]) # middle
    fingers.append(pts[16][1] < pts[14][1]) # ring
    fingers.append(pts[20][1] < pts[18][1]) # pinky
    thumb = pts[4][0] > pts[3][0]           # thumb (x-axis)
    return thumb, fingers

def index_up_only(hand_lm):
    thumb, fingers = finger_states(hand_lm)
    return fingers[0] and not fingers[1] and not fingers[2] and not fingers[3]

def get_gesture(hand_lm):
    thumb, fingers = finger_states(hand_lm)
    if all(fingers) and thumb: return "Palm"
    if not any(fingers) and not thumb: return "Fist"
    return "Other"

# ====================
# Shape helper methods
# ====================
def polygon_angles(poly):
    v = np.array(poly, dtype=np.float32)
    n = len(v); angs = []
    for i in range(n):
        a = v[(i-1)%n] - v[i]; b = v[(i+1)%n] - v[i]
        la = np.linalg.norm(a); lb = np.linalg.norm(b)
        if la*lb == 0: angs.append(0); continue
        cosang = np.clip(np.dot(a,b)/(la*lb), -1, 1)
        angs.append(180 - math.degrees(math.acos(cosang)))
    return angs

def parallel_ratio(poly):
    v = np.array(poly, dtype=np.float32)
    def dir(i):
        d = v[(i+1)%len(v)] - v[i]; n = np.linalg.norm(d)
        return d/n if n>1e-6 else d
    if len(v) < 4: return 0
    d0, d1, d2, d3 = dir(0), dir(1), dir(2), dir(3)
    p1 = abs(np.dot(d0, d2)); p2 = abs(np.dot(d1, d3))
    return (p1+p2)/2

def circularity(contour):
    peri = cv2.arcLength(contour, True)
    if peri == 0: return 0
    area = cv2.contourArea(contour)
    return (4*np.pi*area)/(peri*peri)

def solidity(contour):
    hull = cv2.convexHull(contour)
    ah = cv2.contourArea(hull)
    if ah == 0: return 0
    return cv2.contourArea(contour)/ah

def best_fit_ellipse_ratio(contour):
    try:
        if len(contour) < 5: return 1.0
        (_, _),(MA, ma),_ = cv2.fitEllipse(contour)
        if ma == 0: return 1.0
        return (MA/ma) if MA>ma else (ma/MA)
    except:
        return 1.0

def hu_moments(contour):
    hu = cv2.HuMoments(cv2.moments(contour)).flatten()
    return -np.sign(hu)*np.log10(np.abs(hu)+1e-12)

def hu_distance(h1, h2): return np.linalg.norm(h1 - h2, ord=1)

# ===========
# Templates
# ===========
def template_canvas(draw_fn, size=256, thickness=8):
    img = np.zeros((size,size), np.uint8)
    draw_fn(img, size, thickness)
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return None
    return max(cnts, key=cv2.contourArea)

def draw_star(img, S, t):
    cx, cy, R, r = S//2, S//2, int(S*0.38), int(S*0.15)
    pts = []
    for i in range(10):
        ang = i*36*np.pi/180.0 - np.pi/2
        rad = R if i%2==0 else r
        x = int(cx + rad*np.cos(ang)); y = int(cy + rad*np.sin(ang))
        pts.append((x,y))
    cv2.polylines(img, [np.array(pts,np.int32)], True, 255, t)

def draw_heart(img, S, t):
    cx, cy = S//2, int(S*0.52)
    pts = []
    for a in np.linspace(0, 2*np.pi, 400):
        x = 16*np.sin(a)**3
        y = 13*np.cos(a) - 5*np.cos(2*a) - 2*np.cos(3*a) - np.cos(4*a)
        pts.append((int(cx + x*6), int(cy - y*6)))
    cv2.polylines(img, [np.array(pts,np.int32)], True, 255, t)

def draw_arrow(img, S, t):
    x0, y0 = int(S*0.2), S//2
    shaft = np.array([[x0, y0],[int(S*0.65), y0]], np.int32)
    head  = np.array([[int(S*0.65), int(S*0.35)],
                      [int(S*0.85), y0],
                      [int(S*0.65), int(S*0.65)]], np.int32)
    cv2.polylines(img, [shaft], False, 255, t)
    cv2.polylines(img, [head], True, 255, t)

def draw_plus(img, S, t):
    s = int(S*0.15); cx, cy = S//2, S//2
    cv2.line(img,(cx-s,cy),(cx+s,cy),255,t)
    cv2.line(img,(cx,cy-s),(cx,cy+s),255,t)

def draw_cross(img, S, t):
    s = int(S*0.25); cx, cy = S//2, S//2
    cv2.line(img,(cx-s,cy-s),(cx+s,cy+s),255,t)
    cv2.line(img,(cx-s,cy+s),(cx+s,cy-s),255,t)

def draw_infinity(img, S, t):
    cx, cy = S//2, S//2
    a = int(S*0.18); b = int(S*0.10)
    pts=[]
    for th in np.linspace(0,2*np.pi,600):
        x = a*np.cos(th)/(1+np.sin(th)**2)
        y = b*np.sin(th)*np.cos(th)/(1+np.sin(th)**2)
        pts.append((int(cx + x*2), int(cy + y*2)))
    cv2.polylines(img,[np.array(pts,np.int32)],True,255,t)

TEMPLATES = {}
def build_templates():
    defs = {"Star":draw_star, "Heart":draw_heart, "Arrow":draw_arrow,
            "Plus":draw_plus, "Cross":draw_cross, "Infinity":draw_infinity}
    for name, fn in defs.items():
        cnt = template_canvas(fn)
        if cnt is not None:
            TEMPLATES[name] = hu_moments(cnt)
build_templates()

# ============================================
# Build clean contour from drawn points (key!)
# ============================================
def points_to_clean_contour(points, frame_shape):
    """
    1) Draw polyline into a blank mask
    2) Thicken + close gaps (morphology)
    3) Extract largest contour
    """
    if len(points) < 16: return None
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), np.uint8)

    pts = np.array(points, np.int32)
    cv2.polylines(mask, [pts], isClosed=True, color=255, thickness=6)

    # Close tiny gaps and smooth edges
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8), iterations=2)
    mask = cv2.GaussianBlur(mask, (5,5), 0)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)

    if cv2.contourArea(c) < 200:  # too small
        return None
    return c

# =================================
# Ultra-accurate multi-cue detector
# =================================
def detect_shape_ultra(points, frame_shape):
    c = points_to_clean_contour(points, frame_shape)
    if c is None: return None, {}

    circ = circularity(c)
    sol  = solidity(c)
    ellr = best_fit_ellipse_ratio(c)
    hu   = hu_moments(c)

    # Template (heart/star/arrow/plus/cross/infinity)
    best_name, best_dist = None, 1e9
    for name, temphu in TEMPLATES.items():
        d = hu_distance(hu, temphu)
        if d < best_dist: best_name, best_dist = name, d
    if best_dist < 3.8:  # slightly relaxed
        return best_name, {"circularity":round(circ,3),"solidity":round(sol,3),"ellipse_ratio":round(ellr,3),"hu_dist":round(best_dist,3)}

    # Circle / Oval
    if circ > 0.80 and sol > 0.90:
        return "Circle", {"circularity":round(circ,3),"solidity":round(sol,3),"ellipse_ratio":round(ellr,3)}
    if circ > 0.55 and ellr >= 1.25:
        return "Oval", {"circularity":round(circ,3),"solidity":round(sol,3),"ellipse_ratio":round(ellr,3)}

    # Robust polygon approximation – try multiple epsilons and pick the most stable (lowest std of side lengths)
    peri = cv2.arcLength(c, True)
    best = None
    for factor in [0.015, 0.02, 0.03, 0.04, 0.05]:
        approx = cv2.approxPolyDP(c, factor*peri, True).reshape(-1,2)
        if len(approx) < 3: continue
        # stability score: std of edge lengths / mean
        edges = np.linalg.norm(np.roll(approx,-1,axis=0) - approx, axis=1)
        if edges.mean() == 0: continue
        score = edges.std() / edges.mean()
        if (best is None) or (score < best[0]): best = (score, approx)
    if best is None: return "Unknown", {"circularity":round(circ,3)}

    approx = best[1]; v = len(approx)
    angs = polygon_angles(approx)
    info = {"verts":v, "edge_stability":round(best[0],3), "circ":round(circ,3)}

    if v == 3: return "Triangle", info
    if v == 4:
        pr = parallel_ratio(approx)
        x,y,w,h = cv2.boundingRect(approx); ar = (w/float(h)) if h>0 else 1
        rect = cv2.minAreaRect(c); ang = rect[2]
        # Diamond if rotated ~45° and near-square
        if 0.85 <= ar <= 1.18 and (30 < abs(ang) < 60): return "Diamond", {**info, "ar":round(ar,2),"angle":round(ang,1)}
        if pr > 0.92:
            if 0.80 <= ar <= 1.25: return "Square", {**info, "ar":round(ar,2)}
            else: return "Rectangle", {**info, "ar":round(ar,2)}
        if pr > 0.65: return "Parallelogram", {**info, "ar":round(ar,2)}
        return "Trapezium", {**info, "ar":round(ar,2)}
    if v == 5: return "Pentagon", info
    if v == 6: return "Hexagon", info
    if v == 7: return "Heptagon", info
    if v == 8: return "Octagon", info

    # fallback
    return "Unknown", info

# =========================
# 3D rendering (wireframes)
# =========================
def draw_3d(frame, label):
    h, w, _ = frame.shape
    cx, cy = w//2, h//2

    def cube(x, y, s=160, o=55, col=(0,255,0)):
        cv2.rectangle(frame, (x, y), (x+s, y+s), col, 2)
        cv2.rectangle(frame, (x+o, y-o), (x+s+o, y+s-o), col, 2)
        cv2.line(frame, (x, y), (x+o, y-o), col, 2)
        cv2.line(frame, (x+s, y), (x+s+o, y-o), col, 2)
        cv2.line(frame, (x, y+s), (x+o, y+s-o), col, 2)
        cv2.line(frame, (x+s, y+s), (x+s+o, y+s-o), col, 2)

    def prism(points, top_offset=(45,-45), col=(0,255,255)):
        pts = np.array(points, np.int32)
        cv2.polylines(frame, [pts], True, col, 2)
        pts2 = pts + np.array(top_offset, np.int32)
        cv2.polylines(frame, [pts2], True, col, 2)
        for (p1, p2) in zip(pts, pts2):
            cv2.line(frame, tuple(p1), tuple(p2), col, 2)

    if label in ["Square", "Rectangle", "Diamond", "Parallelogram", "Trapezium"]:
        cube(cx-200, cy-140)
    elif label in ["Triangle", "Pentagon", "Hexagon", "Heptagon", "Octagon"]:
        R = 130
        nmap = {"Triangle":3, "Pentagon":5, "Hexagon":6, "Heptagon":7, "Octagon":8}
        n = nmap[label]
        poly = []
        for i in range(n):
            ang = -np.pi/2 + 2*np.pi*i/n
            poly.append([cx+int(R*np.cos(ang)), cy+int(R*np.sin(ang))])
        prism(poly)
    elif label in ["Circle","Oval"]:
        cv2.circle(frame, (cx, cy), 120, (255,0,0), 3)
        cv2.ellipse(frame, (cx, cy), (120, 38), 0, 0, 360, (255,0,0), 2)
    else:
        # placeholder for extruded complex shapes
        cv2.putText(frame, f"3D {label} (extruded)", (cx-230, cy+180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

# =========================
# App state with dwell start
# =========================
points = deque(maxlen=10000)
confirmed_shape = None
shape_info = {}

HOVER_FRAMES_TO_START = 12   # ~0.4s at ~30fps
HOVER_MOVEMENT_PX   = 10

hover_counter = 0
drawing_mode = False
last_pos = None

# =========================
# Main loop
# =========================
cap = cv2.VideoCapture(0)
print("✅ Index up → hover to start | ✊ Fist → lock to 3D | ✋ Palm → clear")

while True:
    ok, frame = cap.read()
    if not ok: break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    h, w, _ = frame.shape

    if res.multi_hand_landmarks:
        hand_lm = res.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

        ix = int(hand_lm.landmark[8].x * w)
        iy = int(hand_lm.landmark[8].y * h)
        sx, sy = kf.update(ix, iy)

        gesture = get_gesture(hand_lm)

        # ✋ Palm → clear
        if gesture == "Palm":
            points.clear(); confirmed_shape = None; shape_info = {}
            drawing_mode = False; hover_counter = 0; last_pos = None

        # ✊ Fist → lock, classify to 3D
        elif gesture == "Fist":
            drawing_mode = False; hover_counter = 0; last_pos = None
            if len(points) > 30:
                label, info = detect_shape_ultra(list(points), frame.shape)
                confirmed_shape = label if label else "Unknown"
                shape_info = info
            points.clear()

        # ☝ Index-up only = READY; start drawing after hover dwell
        elif index_up_only(hand_lm):
            if last_pos is None:
                last_pos = (sx, sy); hover_counter = 1
            else:
                if np.hypot(sx-last_pos[0], sy-last_pos[1]) <= HOVER_MOVEMENT_PX:
                    hover_counter += 1
                else:
                    hover_counter = 1; last_pos = (sx, sy)
            if hover_counter >= HOVER_FRAMES_TO_START:
                drawing_mode = True
        else:
            hover_counter = 0; last_pos = None
            drawing_mode = False

        # Collect points while drawing
        if drawing_mode:
            points.append((sx, sy))
            cv2.circle(frame, (sx, sy), 5, (255,0,0), -1)
            cv2.putText(frame, "DRAWING...", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
        elif index_up_only(hand_lm) and hover_counter>0:
            rem = max(0, HOVER_FRAMES_TO_START-hover_counter)
            cv2.putText(frame, f"Hold still to start: {rem}", (20,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,200), 2)

    # draw trace
    for p in points:
        cv2.circle(frame, p, 2, (255, 0, 0), -1)

    # draw 3D if confirmed
    if confirmed_shape:
        draw_3d(frame, confirmed_shape)
        cv2.putText(frame, f"{confirmed_shape}", (20, 440),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
        # show a couple debug numbers to verify detection
        y0 = 470
        for k, v in list(shape_info.items())[:3]:
            cv2.putText(frame, f"{k}:{v}", (20, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 2)
            y0 += 25

    cv2.putText(frame, "✋ Palm=Clear  ✊ Fist=Lock  ☝ Index=Hover→Start",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 2)

    cv2.imshow("Ultra-Accurate Air Drawing + All Shapes", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
