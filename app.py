import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import math
import numpy as np

st.set_page_config(page_title="🐵 Real-time Face + Hand Meme", layout="wide")
st.title("🐵 Real-time Face + Hand Expression Meme Display")

# ---------------- Mediapipe Setup ----------------
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

MOUTH_OPEN_THRESHOLD = 0.045
HAND_FACE_DISTANCE_THRESHOLD = 0.08


def euclidean(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def mouth_open_ratio(landmarks):
    top = landmarks.landmark[13]
    bottom = landmarks.landmark[14]
    return abs(top.y - bottom.y)


def detect_expression(face_landmarks, hand_landmarks):
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


# ---------------- Video Transformer ----------------
class MemeVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            max_num_faces=1
        )
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.memes = {
            "serious": cv2.imread("assets/the-monkey-serious-meme.png"),
            "shy": cv2.imread("assets/the-monkey-shy-meme.png"),
            "surprised": cv2.imread("assets/the-monkey-surprised-meme.png"),
            "thinking": cv2.imread("assets/the-monkey-thinking-meme.png")
        }

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_results = self.face_mesh.process(rgb)
        hand_results = self.hands.process(rgb)

        face_landmarks = face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None
        hand_landmarks = hand_results.multi_hand_landmarks if hand_results.multi_hand_landmarks else None

        expression = detect_expression(face_landmarks, hand_landmarks)

        # แสดงชื่ออารมณ์
        cv2.putText(img, f"Expression: {expression.upper()}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # แสดง meme ด้านข้าง
        meme = self.memes.get(expression)
        if meme is not None:
            meme_resized = cv2.resize(meme, (250, 250))
            h, w, _ = img.shape
            img[20:270, w - 270:w - 20] = meme_resized

        return img


# ---------------- Streamlit WebRTC ----------------
webrtc_streamer(
    key="face-hand-meme",
    video_processor_factory=MemeVideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
)
