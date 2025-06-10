import pickle
import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
)
from PySide6.QtGui import QPixmap, QImage, QFont
from PySide6.QtCore import QTimer, Qt
import sys
from datetime import datetime, timedelta


class RollingPrediction:
    def __init__(self, duration_threshold=1.5):
        self.last_char = None
        self.start_time = None
        self.duration_threshold = timedelta(seconds=duration_threshold)

    def update(self, current_char):
        now = datetime.now()
        if current_char != self.last_char:
            self.last_char = current_char
            self.start_time = now
            return None
        elif now - self.start_time >= self.duration_threshold:
            self.start_time = now  # Reset timer to avoid duplicate
            return current_char
        return None


class VocaBotApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VöcaBot - Inference")
        self.setStyleSheet("background-color: white;")
        self.setFixedSize(1100, 750)

        self.camera_on = True
        self.text_accumulated = ""

        self.model = pickle.load(open("model.p", "rb"))["model"]
        self.labels_dict = {i: chr(65 + i) for i in range(26)}  # A-Z

        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)

        self.prediction_tracker = RollingPrediction()

        self.init_ui()
        self.start_timer()

    def init_ui(self):
        sidebar = QVBoxLayout()
        sidebar.setSpacing(30)
        sidebar.setAlignment(Qt.AlignTop)

        def icon_button(text, color):
            btn = QPushButton(text)
            btn.setFixedSize(60, 60)
            btn.setStyleSheet(
                f"border-radius: 30px; font-size: 24px; background-color: {color}; color: white;")
            return btn

        sidebar.addWidget(icon_button("🏠", "#F17022"))
        sidebar.addWidget(icon_button("📷", "#65CDF9"))
        sidebar.addWidget(icon_button("⚙️", "#F17022"))

        title = QLabel("VöcaB🤖t")
        title.setFont(QFont("Arial", 28, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)

        self.video_label = QLabel()
        self.video_label.setFixedSize(700, 400)
        self.video_label.setStyleSheet("border: 2px solid black;")

        self.accumulated_label = QLabel("Accumulated: ")
        self.accumulated_label.setFont(QFont("Arial", 14))
        self.accumulated_label.setStyleSheet("color: black;")
        self.accumulated_label.setFixedWidth(700)

        note = QLabel("Note: This is where the translated text from the customer appear.\n"
                      "Enable and disable button are for turning on and off the camera")
        note.setFont(QFont("Arial", 10))
        note.setStyleSheet(
            "border-top: 2px solid black; border-bottom: 2px solid black; padding: 6px;")
        note.setFixedWidth(700)
        note.setWordWrap(True)

        controls = QHBoxLayout()
        self.enable_btn = QPushButton("Enable")
        self.disable_btn = QPushButton("Disable")
        self.speak_btn = QPushButton("Speak")
        self.reset_btn = QPushButton("Reset")
        for btn in [self.enable_btn, self.disable_btn, self.speak_btn, self.reset_btn]:
            btn.setFixedSize(100, 35)
            btn.setStyleSheet("border: 2px solid black; font-weight: bold;")
        self.enable_btn.clicked.connect(self.enable_camera)
        self.disable_btn.clicked.connect(self.disable_camera)
        self.speak_btn.clicked.connect(self.speak_text)
        self.reset_btn.clicked.connect(self.reset_text)
        controls.addWidget(self.enable_btn)
        controls.addWidget(self.disable_btn)
        controls.addWidget(self.speak_btn)
        controls.addWidget(self.reset_btn)
        controls.setAlignment(Qt.AlignRight)

        right = QVBoxLayout()
        right.addWidget(title)
        right.addWidget(self.video_label)
        right.addWidget(self.accumulated_label)
        right.addWidget(note)
        right.addLayout(controls)
        right.setAlignment(Qt.AlignTop)
        right.setSpacing(15)

        main = QHBoxLayout()
        main.addLayout(sidebar)
        main.addLayout(right)
        self.setLayout(main)

    def start_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def enable_camera(self):
        self.camera_on = True

    def disable_camera(self):
        self.camera_on = False

    def speak_text(self):
        if self.text_accumulated:
            self.engine.say(self.text_accumulated)
            self.engine.runAndWait()

    def reset_text(self):
        self.text_accumulated = ""
        self.accumulated_label.setText("Accumulated: ")

    def update_frame(self):
        if not self.camera_on:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        data_aux, x_, y_ = [], [], []

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            self.mp_drawing.draw_landmarks(frame, hand,
                                           self.mp_hands.HAND_CONNECTIONS,
                                           self.mp_styles.get_default_hand_landmarks_style(),
                                           self.mp_styles.get_default_hand_connections_style())

            for lm in hand.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            if len(data_aux) == 42:
                try:
                    prediction = self.model.predict([np.asarray(data_aux)])
                    index = int(prediction[0])
                    current_char = self.labels_dict.get(index, '?')
                    result = self.prediction_tracker.update(current_char)
                    if result and (not self.text_accumulated or self.text_accumulated[-1] != result):
                        self.text_accumulated += result
                        self.accumulated_label.setText(
                            f"Accumulated: {self.text_accumulated}")
                except Exception as e:
                    print(f"Prediction error: {e}")

        cv2.drawMarker(frame, (W // 2, H // 2),
                       (0, 0, 0), cv2.MARKER_CROSS, 40, 2)

        img = QImage(frame.data, W, H, QImage.Format_RGB888).rgbSwapped()
        self.video_label.setPixmap(QPixmap.fromImage(img))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VocaBotApp()
    win.show()
    sys.exit(app.exec())
