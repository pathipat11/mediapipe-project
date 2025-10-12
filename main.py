"""
Tongue Detection Meme Display
A MediaPipe + OpenCV application that detects when your tongue is out
and displays different meme images accordingly.

Author: Your Name
Tutorial: See TUTORIAL.md for detailed explanations
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import math

# ============================================================================
# CONFIGURATION SETTINGS
# ============================================================================

# Window settings - approximately half monitor size (1920x1080 / 2)
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720

MOUTH_OPEN_THRESHOLD = 0.045
HAND_FACE_DISTANCE_THRESHOLD = 0.08

# Tongue detection threshold - adjust this value to change sensitivity
# Higher value = less sensitive (requires wider mouth opening)
# Lower value = more sensitive (detects smaller mouth openings)
# Recommended range: 0.02 - 0.05
# TONGUE_OUT_THRESHOLD = 0.03

# ============================================================================
# MEDIAPIPE INITIALIZATION
# ============================================================================

# Initialize MediaPipe Face Mesh
# This creates a face detection model that tracks 468 facial landmarks
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    max_num_faces=1
)

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def euclidean(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def mouth_open_ratio(landmarks):
    top = landmarks.landmark[13]
    bottom = landmarks.landmark[14]
    return abs(top.y - bottom.y)

def get_landmark_point(landmarks, idx):
    return landmarks.landmark[idx]

def detect_expression(face_landmarks, hand_landmarks):
    """
    คืนค่าชื่อ expression จากการตรวจจับทั้งใบหน้าและมือ
    - serious: หน้านิ่ง
    - shy: มือยกนิ้วชี้แตะปาก
    - thinking: มือยกนิ้วชี้ขึ้น (เหนือระดับหัว)
    - surprised: ปากอ้า
    """
    if not face_landmarks:
        return "serious"

    # --- 1. ตรวจปากอ้า ---
    mouth_open = mouth_open_ratio(face_landmarks)
    if mouth_open > MOUTH_OPEN_THRESHOLD:
        return "surprised"

    # --- 2. ตรวจมือ ---
    if hand_landmarks:
        for hand in hand_landmarks:
            index_tip = hand.landmark[8]  # ปลายนิ้วชี้

            # จุดสำคัญบนหน้า
            mouth_center = face_landmarks.landmark[13]
            forehead = face_landmarks.landmark[10]
            temple = face_landmarks.landmark[127]  # ข้างหัว

            dist_mouth = euclidean(index_tip, mouth_center)
            dist_forehead = euclidean(index_tip, forehead)

            #  shy: นิ้วใกล้ปาก
            if dist_mouth < HAND_FACE_DISTANCE_THRESHOLD:
                return "shy"

            #  thinking: นิ้วอยู่เหนือหัว (ยกขึ้น)
            if index_tip.y < forehead.y:  # y น้อยกว่า = สูงกว่า
                return "thinking"

    # --- 3. ค่าเริ่มต้น (หน้านิ่ง) ---
    return "serious"


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("="*60)
    print("Monkey Expression Meme Detector 🐵 (Face + Hand Mode)")
    print("="*60)

    meme_files = {
        "serious": "assets/the-monkey-serious-meme.png",
        "shy": "assets/the-monkey-shy-meme.png",
        "surprised": "assets/the-monkey-surprised-meme.png",
        "thinking": "assets/the-monkey-thinking-meme.png"
    }

    memes = {}
    for key, path in meme_files.items():
        if not os.path.exists(path):
            print(f"[ERROR] Missing image: {path}")
            return
        img = cv2.imread(path)
        memes[key] = cv2.resize(img, (WINDOW_WIDTH, WINDOW_HEIGHT))
    print("[OK] Meme images loaded.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    cv2.namedWindow("Camera Input", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Meme Output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Input", WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.resizeWindow("Meme Output", WINDOW_WIDTH, WINDOW_HEIGHT)

    current_meme = memes["serious"].copy()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # process ทั้งหน้าและมือ
        face_results = face_mesh.process(rgb)
        hand_results = hands.process(rgb)

        face_landmarks = face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None
        hand_landmarks = hand_results.multi_hand_landmarks if hand_results.multi_hand_landmarks else None

        expression = detect_expression(face_landmarks, hand_landmarks)
        current_meme = memes[expression].copy()

        # วาด landmarks เพื่อ debug
        if hand_landmarks:
            for hand in hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        if face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                        landmark_drawing_spec=None,
                                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))

        cv2.putText(frame, f"Expression: {expression}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Camera Input", frame)
        cv2.imshow("Meme Output", current_meme)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    hands.close()
    print("[OK] Application closed successfully.")

if __name__ == "__main__":
    main()