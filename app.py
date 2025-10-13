import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
import os

# =====================================================================
# MediaPipe Setup
# =====================================================================
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

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

# =====================================================================
# Constants
# =====================================================================
MOUTH_OPEN_THRESHOLD = 0.045
HAND_FACE_DISTANCE_THRESHOLD = 0.08

# =====================================================================
# Helper Functions
# =====================================================================
def euclidean(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def mouth_open_ratio(landmarks):
    """Calculate the distance between top and bottom lip."""
    top = landmarks.landmark[13]
    bottom = landmarks.landmark[14]
    return abs(top.y - bottom.y)

def detect_expression_from_image(image):
    """Detect facial expression or hand gesture from an image."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(rgb)
    hand_results = hands.process(rgb)

    face_landmarks = face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None
    hand_landmarks = hand_results.multi_hand_landmarks if hand_results.multi_hand_landmarks else None

    if not face_landmarks:
        return "serious"

    mouth_open = mouth_open_ratio(face_landmarks)
    if mouth_open > MOUTH_OPEN_THRESHOLD:
        return "surprised"

    if hand_landmarks:
        for hand in hand_landmarks:
            index_tip = hand.landmark[8]
            mouth_center = face_landmarks.landmark[13]
            forehead = face_landmarks.landmark[10]

            dist_mouth = euclidean(index_tip, mouth_center)
            if dist_mouth < HAND_FACE_DISTANCE_THRESHOLD:
                return "shy"

            if index_tip.y < forehead.y:
                return "thinking"

    return "serious"

# =====================================================================
# Streamlit UI
# =====================================================================
st.set_page_config(page_title="🐵 Face + Hand Expression Meme", layout="centered")
st.title("🐵 Face + Hand Expression Meme Display")

st.write("Upload a photo and I’ll guess your **expression** — then show the matching meme!")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Load meme assets
meme_files = {
    "serious": "assets/the-monkey-serious-meme.png",
    "shy": "assets/the-monkey-shy-meme.png",
    "surprised": "assets/the-monkey-surprised-meme.png",
    "thinking": "assets/the-monkey-thinking-meme.png"
}

memes = {}
for key, path in meme_files.items():
    if os.path.exists(path):
        memes[key] = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    else:
        memes[key] = np.zeros((300, 300, 3), dtype=np.uint8)  # fallback black image

if uploaded:
    # Convert uploaded file → OpenCV image
    bytes_data = uploaded.read()
    image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Detect expression
    expression = detect_expression_from_image(image)

    st.subheader(f"😄 Detected Expression: **{expression.upper()}**")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

    if expression in memes:
        st.image(memes[expression], caption=f"Meme for {expression}", use_column_width=True)
    else:
        st.warning("No meme image found for this expression.")
else:
    st.info("👆 Upload an image above to start!")
