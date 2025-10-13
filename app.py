import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import mediapipe as mp
import math
import numpy as np
import time

st.set_page_config(page_title="Real-time Face + Hand Meme", layout="wide")
st.title("Real-time Face + Hand Expression Meme Display")

# ---------------- Mediapipe Setup ----------------
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

MOUTH_OPEN_THRESHOLD = 0.045
HAND_FACE_DISTANCE_THRESHOLD = 0.08

# ---------------- Streamlit Controls ----------------
show_overlay = st.sidebar.checkbox("Show Overlay (lines & distances)", value=True)

# ---------------- Helper Functions ----------------
def euclidean(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def mouth_open_ratio(landmarks):
    top = landmarks.landmark[13]
    bottom = landmarks.landmark[14]
    return abs(top.y - bottom.y)

def detect_expression(face_landmarks, hand_landmarks, frame=None, overlay=True):
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
            dist_forehead = euclidean(index_tip, forehead)

            if overlay and frame is not None:
                h, w, _ = frame.shape
                p_index = (int(index_tip.x * w), int(index_tip.y * h))
                p_mouth = (int(mouth_center.x * w), int(mouth_center.y * h))
                p_forehead = (int(forehead.x * w), int(forehead.y * h))

                # Draw points
                cv2.circle(frame, p_index, 8, (255, 0, 0), -1)
                cv2.circle(frame, p_mouth, 6, (0, 255, 0), -1)
                cv2.circle(frame, p_forehead, 6, (0, 255, 255), -1)

                # Draw lines
                cv2.line(frame, p_index, p_mouth, (0, 255, 0), 2)
                cv2.line(frame, p_index, p_forehead, (0, 255, 255), 2)

                # Show distances
                cv2.putText(frame, f"mouth:{dist_mouth:.3f}", (p_index[0]+10, p_index[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(frame, f"forehead:{dist_forehead:.3f}", (p_index[0]+10, p_index[1]+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            # Expression detection
            if dist_mouth < HAND_FACE_DISTANCE_THRESHOLD:
                return "shy"
            if index_tip.y < forehead.y:
                return "thinking"

    return "serious"

# ---------------- Video Transformer ----------------
class MemeVideoTransformer(VideoTransformerBase):
    def __init__(self, overlay=True):
        self.overlay = overlay
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

        # โหลด meme
        self.memes = {
            "serious": cv2.imread("assets/the-monkey-serious-meme.png"),
            "shy": cv2.imread("assets/the-monkey-shy-meme.png"),
            "surprised": cv2.imread("assets/the-monkey-surprised-meme.png"),
            "thinking": cv2.imread("assets/the-monkey-thinking-meme.png")
        }
        for k in self.memes:
            if self.memes[k] is None:
                self.memes[k] = np.zeros((250, 250, 3), dtype=np.uint8)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_results = self.face_mesh.process(rgb)
        hand_results = self.hands.process(rgb)

        face_landmarks = face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None
        hand_landmarks = hand_results.multi_hand_landmarks if hand_results.multi_hand_landmarks else None

        expression = detect_expression(face_landmarks, hand_landmarks, img, overlay=self.overlay)

        # วาด face & hand landmarks
        if self.overlay:
            if face_landmarks:
                mp_drawing.draw_landmarks(
                    img, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
                )

            if hand_landmarks:
                for hand in hand_landmarks:
                    mp_drawing.draw_landmarks(
                        img, hand, mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=3)
                    )

        # แสดงชื่ออารมณ์
        cv2.putText(img, f"{expression.upper()}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        self.current_expression = expression
        self.current_frame = img
        return img

# ---------------- WebRTC Configuration ----------------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        # STUN Servers
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:global.stun.twilio.com:3478"]},
        # TURN Server สาธารณะ (ตัวเลือกสำรองที่เพิ่มความหลากหลาย)
        {"urls": "turn:openrelay.metered.ca:80", "username": "openrelayproject", "credential": "openrelayproject"},
        {"urls": "turn:openrelay.metered.ca:443", "username": "openrelayproject", "credential": "openrelayproject"},
    ],
    # *** เพิ่มบรรทัดนี้เพื่อบังคับใช้ TCP ***
    "iceTransportPolicy": "relay", # บังคับให้ใช้ TURN/Relay
    # อีกตัวเลือกคือ "all" หรือ "public", แต่ "relay" เป็นการทดสอบ TURN/TCP ได้ดีที่สุด
    } 
)

# ---------------- Layout ----------------
col1, col2 = st.columns([2, 1])

with col1:
    webrtc_ctx = webrtc_streamer(
        key="face-hand-meme",
        video_processor_factory=lambda: MemeVideoTransformer(overlay=show_overlay),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration=RTC_CONFIGURATION, 
    )

with col2:
    st.subheader("Meme Output")
    meme_placeholder = st.empty()

# ---------------- Real-time Meme Update ----------------
while True:
    time.sleep(0.1)
    if webrtc_ctx.video_processor:
        # --- Update overlay real-time ---
        webrtc_ctx.video_processor.overlay = show_overlay
        expression = getattr(webrtc_ctx.video_processor, "current_expression", None)
        frame = getattr(webrtc_ctx.video_processor, "current_frame", None)
        if expression and frame is not None:
            meme = webrtc_ctx.video_processor.memes.get(expression)
            if meme is not None:
                meme_placeholder.image(
                    cv2.cvtColor(meme, cv2.COLOR_BGR2RGB),
                    caption=f"Meme: {expression.upper()}",
                    use_container_width=True
                )
