import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np
import pickle
import threading
import pyttsx3
import time

# Load ML model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
labels_dict = [chr(65 + i) for i in range(26)]

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Text-to-Speech
engine = pyttsx3.init()

# Tkinter UI setup
root = tk.Tk()
root.title("VöcaBot")
root.geometry("960x540")
root.configure(bg="white")


def load_icon(path, size=(40, 40)):
    img = Image.open(path)
    img = img.resize(size, Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(img)


tk.Label(root, text="VöcaBot", font=("Helvetica", 28, "bold"),
         bg="white").pack(pady=(20, 10))

main_frame = tk.Frame(root, bg="white")
main_frame.pack(fill="both", expand=True, padx=20, pady=10)

side_frame = tk.Frame(main_frame, bg="white")
side_frame.pack(side="left", padx=20, fill="y")

home_icon = load_icon("icons/home.png")
camera_icon = load_icon("icons/camera.png")
settings_icon = load_icon("icons/gear.png")

tk.Button(side_frame, image=home_icon, bd=0, bg="white").pack(pady=20)
tk.Button(side_frame, image=camera_icon, bd=0, bg="white").pack(pady=20)
tk.Button(side_frame, image=settings_icon, bd=0, bg="white").pack(pady=20)

camera_frame = tk.Frame(main_frame, bg="white", bd=2, relief="solid")
camera_frame.pack(side="left", expand=True, fill="both", padx=10, pady=10)

camera_canvas = tk.Canvas(camera_frame, bg="white", highlightthickness=0)
camera_canvas.pack(fill="both", expand=True)
camera_canvas.image_id = None

bottom_frame = tk.Frame(root, bg="white")
bottom_frame.pack(pady=10)

# Dynamic detected text display
transcribed_text = tk.StringVar()
transcribed_text.set("Detected Text: ")
output_label = tk.Label(bottom_frame, textvariable=transcribed_text,
                        font=("Helvetica", 14), bg="white", fg="black", wraplength=800, justify="left")
output_label.pack(pady=5)

button_frame = tk.Frame(bottom_frame, bg="white")
button_frame.pack(pady=5)

# Controls state
cap = None
running = False
last_prediction = None
last_time = 0
word_buffer = ""


def update_display():
    transcribed_text.set("Detected Text: " + word_buffer)


def backspace():
    global word_buffer
    word_buffer = word_buffer[:-1]
    update_display()


def speak_word():
    if word_buffer:
        engine.say(word_buffer)
        engine.runAndWait()


def predict_and_display():
    global cap, running, last_prediction, last_time, word_buffer
    cap = cv2.VideoCapture(0)

    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W, _ = frame.shape

        results = hands.process(frame_rgb)
        predicted_character = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks,
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

                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
                break

        current_time = time.time()
        if predicted_character:
            if predicted_character == last_prediction:
                if current_time - last_time >= 1.5:
                    word_buffer += predicted_character
                    update_display()
                    last_time = current_time
            else:
                last_prediction = predicted_character
                last_time = current_time

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)

        canvas_w = camera_canvas.winfo_width()
        canvas_h = camera_canvas.winfo_height()
        x_offset = (canvas_w - W) // 2 if canvas_w > W else 0
        y_offset = (canvas_h - H) // 2 if canvas_h > H else 0

        if camera_canvas.image_id is None:
            camera_canvas.image_id = camera_canvas.create_image(
                x_offset, y_offset, anchor='nw', image=imgtk)
        else:
            camera_canvas.coords(camera_canvas.image_id, x_offset, y_offset)
            camera_canvas.itemconfig(camera_canvas.image_id, image=imgtk)

        camera_canvas.imgtk = imgtk

    cap.release()


def enable_camera():
    global running
    if not running:
        running = True
        threading.Thread(target=predict_and_display, daemon=True).start()


def disable_camera():
    global running
    running = False


ttk.Button(button_frame, text="Enable",
           command=enable_camera).pack(side="left", padx=10)
ttk.Button(button_frame, text="Disable",
           command=disable_camera).pack(side="left", padx=10)
ttk.Button(button_frame, text="Backspace",
           command=backspace).pack(side="left", padx=10)
ttk.Button(button_frame, text="Speak",
           command=speak_word).pack(side="left", padx=10)

root.mainloop()
