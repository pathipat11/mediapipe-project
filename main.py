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
    """Calculate Euclidean distance between two 3D landmarks."""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def mouth_open_ratio(landmarks):
    """Measure mouth opening ratio based on upper and lower lip distance."""
    top = landmarks.landmark[13]
    bottom = landmarks.landmark[14]
    return abs(top.y - bottom.y)

def get_landmark_point(landmarks, idx):
    return landmarks.landmark[idx]

def detect_expression(face_landmarks, hand_landmarks, frame=None, show_drawing=True):
    """
    Detect facial expression or hand gesture and return expression name.
    Optionally draw debug lines if frame is provided.

    - serious: neutral face
    - shy: index finger near mouth
    - thinking: index finger raised above forehead
    - surprised: mouth open
    """
    if not face_landmarks:
        return "serious"

    # --- 1. Detect mouth open ---
    mouth_open = mouth_open_ratio(face_landmarks)
    if mouth_open > MOUTH_OPEN_THRESHOLD:
        return "surprised"

    # --- 2. Detect hand position ---
    if hand_landmarks:
        for hand in hand_landmarks:
            index_tip = hand.landmark[8]

            # Key facial points
            mouth_center = face_landmarks.landmark[13]
            forehead = face_landmarks.landmark[10]

            dist_mouth = euclidean(index_tip, mouth_center)
            dist_forehead = euclidean(index_tip, forehead)

            # --- Debug visualization ---
            if show_drawing and frame is not None:
                h, w, _ = frame.shape
                p_index = (int(index_tip.x * w), int(index_tip.y * h))
                p_mouth = (int(mouth_center.x * w), int(mouth_center.y * h))
                p_forehead = (int(forehead.x * w), int(forehead.y * h))

                # Draw debug points
                cv2.circle(frame, p_index, 8, (255, 0, 0), -1)       # Index finger (blue)
                cv2.circle(frame, p_mouth, 6, (0, 255, 0), -1)       # Mouth (green)
                cv2.circle(frame, p_forehead, 6, (0, 255, 255), -1)  # Forehead (yellow)

                # Draw debug lines
                cv2.line(frame, p_index, p_mouth, (0, 255, 0), 2)
                cv2.line(frame, p_index, p_forehead, (0, 255, 255), 2)

                # Show distance text
                cv2.putText(frame, f"mouth:{dist_mouth:.3f}", (p_index[0] + 10, p_index[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"forehead:{dist_forehead:.3f}", (p_index[0] + 10, p_index[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # ✅ shy: finger near mouth
            if dist_mouth < HAND_FACE_DISTANCE_THRESHOLD:
                return "shy"

            # ✅ thinking: finger above forehead
            if index_tip.y < forehead.y:
                return "thinking"

    # --- 3. Default: neutral ---
    return "serious"


def draw_expression_overlay(frame, expression):
    """Draw a transparent overlay with expression text."""
    overlay = frame.copy()
    h, w, _ = frame.shape

    # Background rectangle (semi-transparent)
    rect_height = 60
    cv2.rectangle(overlay, (0, 0), (w, rect_height), (0, 0, 0), -1)

    # Blend overlay with transparency
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Expression text style
    text = f"Expression: {expression.upper()}"
    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3, cv2.LINE_AA)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    print("=" * 60)
    print("Monkey Expression Meme Detector 🐵 (Face + Hand Mode)")
    print("=" * 60)

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

    draw_enabled = True
    current_meme = memes["serious"].copy()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process face and hand landmarks
        face_results = face_mesh.process(rgb)
        hand_results = hands.process(rgb)

        face_landmarks = face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None
        hand_landmarks = hand_results.multi_hand_landmarks if hand_results.multi_hand_landmarks else None

        expression = detect_expression(face_landmarks, hand_landmarks, frame, show_drawing=draw_enabled)
        current_meme = memes[expression].copy()

        # Draw hand landmarks
        if draw_enabled and hand_landmarks:
            for hand in hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # Draw face landmarks
        if draw_enabled and face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                        landmark_drawing_spec=None,
                                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0),
                                                                                        thickness=1,
                                                                                        circle_radius=1))

        # --- Draw overlay label ---
        if draw_enabled:
            draw_expression_overlay(frame, expression)

        # Display both camera feed and meme output
        cv2.imshow("Camera Input", frame)
        cv2.imshow("Meme Output", current_meme)

        # Press 'q' to exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('e'):
            draw_enabled = not draw_enabled
            print(f"[INFO] Drawing landmarks: {'ON' if draw_enabled else 'OFF'}")
        elif key == ord('q'):
            break


    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    hands.close()
    print("[OK] Application closed successfully.")


if __name__ == "__main__":
    main()