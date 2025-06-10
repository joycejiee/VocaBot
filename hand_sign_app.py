import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

# UI Header
st.title("🤟 Real-time Hand Sign Detector")
st.write("Uses Mediapipe and your trained model to predict hand signs from webcam.")

# Load your model safely
model = None
labels_dict = [chr(65 + i) for i in range(26)]

try:
    with open('./model.p', 'rb') as f:
        model_dict = pickle.load(f)
        model = model_dict['model']
except Exception as e:
    st.error(f"❌ Failed to load model.p: {e}")
    st.stop()

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# WebRTC setup
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})


class HandSignProcessor(VideoProcessorBase):
    def __init__(self):
        self.result_text = ""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        H, W, _ = img.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                x_, y_, data_aux = [], [], []

                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10

                prediction = model.predict([np.asarray(data_aux)])
                predicted_char = labels_dict[int(prediction[0])]

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
                cv2.putText(img, predicted_char, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 255, 0), 3, cv2.LINE_AA)

                self.result_text = predicted_char
                break

        return img


# Start the webcam stream
ctx = webrtc_streamer(
    key="hand-sign-detection",
    mode=WebRtcMode.OPENCV,  # FIXED: use WebRtcMode enum instead of plain string
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=HandSignProcessor,
    media_stream_constraints={"video": True, "audio": False}
)
