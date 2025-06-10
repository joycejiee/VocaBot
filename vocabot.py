import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import time
import pickle
import mediapipe as mp
import pyttsx3
import openai
from collections import deque
from statistics import mode

# ---------------- Setup ---------------- #

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
labels_dict = {i: chr(65 + i) for i in range(26)}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

engine = pyttsx3.init()
engine.setProperty('rate', 150)

openai.api_key = "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"  # Replace with your actual key

cap = cv2.VideoCapture(0)
camera_on = True

letter_buffer = []
confirmed_words = []
stable_letter = None
stable_start_time = None
required_stable_duration = 1.0
letter_window = deque(maxlen=5)

# GPT Suggestion with Confidence


def gpt_suggest_word(confirmed_words, current_letters):
    context = f"Confirmed sentence so far: {' '.join(confirmed_words)}\n"
    prompt = (
        context +
        f"Letters being signed: '{current_letters}'\n"
        "1. What is the most likely word?\n"
        "2. How confident are you (high, medium, low)?\n"
        "Respond with: word | confidence"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=20
        )
        text = response.choices[0].message.content.strip()
        if "|" in text:
            word, confidence = text.split("|")
            return word.strip(), confidence.strip()
        return text.strip(), "unknown"
    except Exception as e:
        print("GPT Error:", e)
        return current_letters, "error"

# Video Loop


def video_loop():
    global letter_buffer, stable_letter, stable_start_time

    if not camera_on:
        return

    ret, frame = cap.read()
    if not ret:
        root.after(10, video_loop)
        return

    data_aux = []
    x_, y_ = [], []
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        for lm in hand_landmarks.landmark:
            x_.append(lm.x)
            y_.append(lm.y)

        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))

        if len(data_aux) == 42:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            letter_window.append(predicted_character)

            if len(letter_window) == letter_window.maxlen:
                try:
                    smooth_letter = mode(letter_window)
                    current_time = time.time()
                    if smooth_letter == stable_letter:
                        if stable_start_time and (current_time - stable_start_time) >= required_stable_duration:
                            if not letter_buffer or letter_buffer[-1] != smooth_letter:
                                letter_buffer.append(smooth_letter)
                                stable_start_time = None
                    else:
                        stable_letter = smooth_letter
                        stable_start_time = current_time
                except:
                    pass

    raw_letters = ''.join(letter_buffer).lower()
    suggested_word, confidence = gpt_suggest_word(confirmed_words, raw_letters)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img).resize((600, 400))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    current_translated.set(
        f"Current: {raw_letters} → {suggested_word}  ({confidence})")
    confirmed_text.set(f"Confirmed: {' '.join(confirmed_words)}")

    root.after(10, video_loop)

# Controls


def start_camera():
    global camera_on
    if not camera_on:
        camera_on = True
        video_loop()


def stop_camera():
    global camera_on
    camera_on = False


def speak_word():
    global letter_buffer, confirmed_words, stable_letter, stable_start_time

    raw_letters = ''.join(letter_buffer).lower()
    suggested_word, _ = gpt_suggest_word(confirmed_words, raw_letters)

    if suggested_word:
        confirmed_words.append(suggested_word)
        engine.say(suggested_word)
        engine.runAndWait()

    letter_buffer.clear()
    stable_letter = None
    stable_start_time = None
    current_translated.set("Current: ")
    confirmed_text.set(f"Confirmed: {' '.join(confirmed_words)}")


def reset_word():
    global confirmed_words, letter_buffer, stable_letter, stable_start_time
    confirmed_words = []
    letter_buffer = []
    stable_letter = None
    stable_start_time = None
    current_translated.set("Current: ")
    confirmed_text.set("Confirmed: ")


# GUI Layout
root = tk.Tk()
root.title("VöcaBot - Enhanced Word Construction")
root.geometry("950x700")
root.configure(bg="white")

tk.Label(root, text="VöcaB🤖t", font=(
    "Helvetica", 26, "bold"), bg="white").pack(pady=10)

main_frame = tk.Frame(root, bg="white")
main_frame.pack(fill="both", expand=True, padx=20)

sidebar = tk.Frame(main_frame, width=100, bg="white")
sidebar.pack(side="left", fill="y", padx=10)


def icon_button(emoji):
    return tk.Button(sidebar, text=emoji, font=("Arial", 20), width=3, height=1, bg="white", bd=0, highlightthickness=0)


icon_button("🏠").pack(pady=10)
icon_button("📷💬").pack(pady=10)
icon_button("⚙️").pack(pady=10)

video_container = tk.Frame(main_frame, width=600, height=400,
                           bg="white", highlightbackground="black", highlightthickness=2)
video_container.pack(side="left", padx=20)

video_label = tk.Label(video_container, bg="white")
video_label.pack(fill="both", expand=True)

current_translated = tk.StringVar()
confirmed_text = tk.StringVar()
current_translated.set("Current: ")
confirmed_text.set("Confirmed: ")

tk.Label(root, textvariable=current_translated, font=(
    "Helvetica", 14), bg="white", fg="blue").pack(pady=(5, 0))
tk.Label(root, textvariable=confirmed_text, font=(
    "Helvetica", 14, "bold"), bg="white", fg="green").pack(pady=(0, 10))

bottom_frame = tk.Frame(root, bg="white")
bottom_frame.pack(fill="x", pady=10, padx=40)

note_text = ("Note: This is where the translated text from the customer appears.\n"
             "Enable and disable buttons are for turning the camera on and off.")
note_label = tk.Label(bottom_frame, text=note_text, font=(
    "Helvetica", 10), bg="white", justify="left")
note_label.pack(anchor="w")

tk.Frame(bottom_frame, bg="black", height=1).pack(fill="x", pady=(2, 0))
tk.Frame(bottom_frame, bg="black", height=1).pack(fill="x", pady=(0, 10))

btn_frame = tk.Frame(bottom_frame, bg="white")
btn_frame.pack(anchor="e")

ttk.Button(btn_frame, text="Enable", command=start_camera).pack(
    side="left", padx=10)
ttk.Button(btn_frame, text="Disable", command=stop_camera).pack(
    side="left", padx=10)
ttk.Button(btn_frame, text="Speak", command=speak_word).pack(
    side="left", padx=10)
ttk.Button(btn_frame, text="Reset", command=reset_word).pack(
    side="left", padx=10)

video_loop()
root.mainloop()

cap.release()
