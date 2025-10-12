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
EYE_CLOSE_THRESHOLD = 0.020
EYEBROW_DIFF_THRESHOLD = 0.015

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
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.6,  # Confidence threshold for initial detection
    min_tracking_confidence=0.6,   # Confidence threshold for tracking
    max_num_faces=1                # We only need to track one face
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

def eye_aspect_ratio(landmarks, ids):
    # ids = (upper_lid, lower_lid)
    return abs(landmarks.landmark[ids[0]].y - landmarks.landmark[ids[1]].y)

def eyebrow_gap(landmarks, eyebrow_id, eye_id):
    return abs(landmarks.landmark[eyebrow_id].y - landmarks.landmark[eye_id].y)

def detect_expression(landmarks):
    mouth_open = mouth_open_ratio(landmarks)
    left_eye = eye_aspect_ratio(landmarks, (159, 145))  # left eye top-bottom
    right_eye = eye_aspect_ratio(landmarks, (386, 374))  # right eye top-bottom

    left_brow_gap = eyebrow_gap(landmarks, 70, 159)
    right_brow_gap = eyebrow_gap(landmarks, 300, 386)

    # Expression logic
    if mouth_open > MOUTH_OPEN_THRESHOLD:
        return "surprised"
    elif left_eye < EYE_CLOSE_THRESHOLD and right_eye < EYE_CLOSE_THRESHOLD:
        return "shy"
    elif abs(left_brow_gap - right_brow_gap) > EYEBROW_DIFF_THRESHOLD:
        return "thinking"
    else:
        return "serious"

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("="*60)
    print("Monkey Expression Meme Detector")
    print("="*60)

    # Load images
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
        print("[ERROR] Cannot access webcam.")
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
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                expression = detect_expression(face_landmarks)
                current_meme = memes[expression].copy()
                cv2.putText(frame, f"Expression: {expression}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        else:
            current_meme = memes["serious"].copy()
            cv2.putText(frame, "No face detected", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Camera Input", frame)
        cv2.imshow("Meme Output", current_meme)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("[OK] Application closed successfully.")

if __name__ == "__main__":
    main()
