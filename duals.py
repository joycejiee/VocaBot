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
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPEN_API_KEY")

# Load ML model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
labels_dict = [chr(65 + i) for i in range(26)]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

engine = pyttsx3.init()
word_buffer = ""
last_prediction = None
last_time = 0
running = False
cap = None

# Main window: Deaf Crew
root = tk.Tk()
root.title("VöcaBot - Crew")
root.geometry("960x540")
root.configure(bg="white")

tk.Label(root, text="VöcaBot", font=("Helvetica", 28, "bold"),
         bg="white").pack(pady=(20, 10))

camera_frame = tk.Frame(root, bg="white", bd=2, relief="solid")
camera_frame.pack(padx=50, fill="x", pady=(0, 10))
camera_canvas = tk.Canvas(camera_frame, bg="white",
                          height=250, highlightthickness=0)
camera_canvas.pack(fill="both", expand=True)
camera_canvas.image_id = None

transcribed_text = tk.StringVar(value="Detected Text:")
tk.Label(root, textvariable=transcribed_text, font=("Helvetica", 12, "bold"),
         bg="white", anchor="w").pack(padx=50, anchor="w")
tk.Frame(root, height=2, bg="black").pack(fill="x", padx=50, pady=(0, 10))

deaf_button_frame = tk.Frame(root, bg="white")
deaf_button_frame.pack(pady=5)


def update_text():
    transcribed_text.set("Detected Text: " + word_buffer)
    customer_note_var.set("Detected Text: " + word_buffer)
    msg_canvas.itemconfig(msg_canvas_text_id, text=word_buffer)


def backspace():
    global word_buffer
    word_buffer = word_buffer[:-1]
    update_text()


def speak_word():
    if word_buffer:
        engine.say(word_buffer)
        engine.runAndWait()


def start_camera():
    global running
    if not running:
        running = True
        threading.Thread(target=predict_and_display, daemon=True).start()


def stop_camera():
    global running
    running = False


for label, cmd in [("Enable", start_camera), ("Disable", stop_camera),
                   ("Backspace", backspace), ("Speak", speak_word)]:
    tk.Button(deaf_button_frame, text=label, bg="#f57c00", fg="white",
              font=("Helvetica", 10, "bold"), padx=10, pady=5, relief="raised",
              command=cmd).pack(side="left", padx=10)

# Customer window
cust_win = tk.Toplevel(root)
cust_win.title("VöcaBot - Speech to Text")
cust_win.geometry("960x540")
cust_win.configure(bg="white")

tk.Label(cust_win, text="VöcaBot", font=(
    "Helvetica", 28, "bold"), bg="white").pack(pady=(20, 10))

msg_frame = tk.Frame(cust_win, bg="white", bd=2, relief="solid")
msg_frame.pack(padx=50, fill="x", pady=(0, 10))
msg_canvas = tk.Canvas(msg_frame, bg="white", height=250, highlightthickness=0)
msg_canvas.pack(fill="both", expand=True)
msg_canvas_text_id = msg_canvas.create_text(
    10, 125, anchor="w", text="", font=("Helvetica", 18))

customer_note_var = tk.StringVar(value="Detected Text:")
tk.Label(cust_win, textvariable=customer_note_var, font=("Helvetica", 12, "bold"),
         bg="white", anchor="w").pack(padx=50, anchor="w")
tk.Frame(cust_win, height=2, bg="black").pack(fill="x", padx=50, pady=(0, 10))

cust_button_frame = tk.Frame(cust_win, bg="white")
cust_button_frame.pack(pady=10, fill="x", padx=50)

speech_var = tk.StringVar()


def send_text():
    global word_buffer
    spoken_text = speech_var.get().strip()
    if spoken_text:
        word_buffer = spoken_text
        update_text()
        engine.say(spoken_text)
        engine.runAndWait()


def reset_text():
    global word_buffer
    word_buffer = ""
    update_text()


def capture_and_transcribe():
    duration = 5
    fs = 44100
    print("🎙️ Recording...")

    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        write(temp_wav.name, fs, recording)

        with open(temp_wav.name, "rb") as audio_file:
            try:
                transcript = openai.Audio.transcribe("whisper-1", audio_file)
                spoken_text = transcript["text"]
                speech_var.set(spoken_text)
                send_text()
            except Exception as e:
                print("❌ Error with transcription:", e)


# Left buttons: Send, Reset, Transcribe
left_btns = tk.Frame(cust_button_frame, bg="white")
left_btns.pack(side="left")

for label, cmd in [("Send", send_text), ("Reset", reset_text), ("🎤 Transcribe", capture_and_transcribe)]:
    tk.Button(left_btns, text=label, bg="#f57c00", fg="white",
              font=("Helvetica", 10, "bold"), padx=10, pady=5, relief="raised",
              command=cmd).pack(side="left", padx=10)

# Right buttons: Enable, Disable
right_btns = tk.Frame(cust_button_frame, bg="white")
right_btns.pack(side="right")

for label, cmd in [("Enable", start_camera), ("Disable", stop_camera)]:
    tk.Button(right_btns, text=label, bg="#f57c00", fg="white",
              font=("Helvetica", 10, "bold"), padx=10, pady=5, relief="raised",
              command=cmd).pack(side="left", padx=10)

# Prediction and Camera Drawing


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
                    last_time = current_time
                    update_text()
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


root.mainloop()
