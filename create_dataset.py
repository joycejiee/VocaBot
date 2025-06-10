import os
import pickle
import mediapipe as mp
import cv2

DATA_DIR = './data/'

data = []
labels = []

mp_hands = mp.solutions.hands

with mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3) as hands:
    for label in os.listdir(DATA_DIR):
        label_path = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_path):
            continue

        print(f"\nProcessing label: {label}")

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)

            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"❌ Could not read {img_path}, skipping...")
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                if results.multi_hand_landmarks:
                    data_aux = []
                    x_ = []
                    y_ = []

                    for landmark in results.multi_hand_landmarks[0].landmark:
                        x_.append(landmark.x)
                        y_.append(landmark.y)

                    for landmark in results.multi_hand_landmarks[0].landmark:
                        data_aux.append(landmark.x - min(x_))
                        data_aux.append(landmark.y - min(y_))

                    data.append(data_aux)
                    labels.append(label)

                    print(f"✅ Processed {img_name}")
                else:
                    print(f"⚠️ No hands detected in {img_name}")

            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")

if data and labels:
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print("\n✅ Dataset saved to data.pickle")
else:
    print("\n⚠️ No valid data found. Dataset not saved.")
