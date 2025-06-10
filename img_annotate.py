import cv2
import os
import numpy as np
from PIL import Image

# Load the ASL chart you uploaded
image_path = 'asl.jpg'  # Use your actual file path
output_dir = 'dataset'
os.makedirs(output_dir, exist_ok=True)

# Alphabet labels
labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# Mouse callback variables
ref_point = []
cropping = False
current_label_index = 0

# Augmentation (rotate, noise)


def augment(img):
    h, w = img.shape[:2]
    # random rotation
    angle = np.random.randint(-10, 10)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    img_rot = cv2.warpAffine(img, M, (w, h))
    # random brightness
    value = np.random.randint(-30, 30)
    hsv = cv2.cvtColor(img_rot, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], value)
    img_bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img_bright

# Save N images per cropped region


def save_cropped_region(crop, label, count=100):
    folder = os.path.join(output_dir, label)
    os.makedirs(folder, exist_ok=True)
    for i in range(count):
        aug = augment(crop)
        filename = os.path.join(folder, f'{label}_{i}.jpg')
        cv2.imwrite(filename, aug)

# Mouse event callback


def click_crop(event, x, y, flags, param):
    global ref_point, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False

        x1, y1 = ref_point[0]
        x2, y2 = ref_point[1]
        roi = clone[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]

        current_label = labels[current_label_index]
        print(f"[✔] Cropped region for '{current_label}'")

        save_cropped_region(roi, current_label)
        next_label()


def next_label():
    global current_label_index
    current_label_index += 1
    if current_label_index < len(labels):
        print(f"[→] Draw box for letter '{labels[current_label_index]}'")
    else:
        print("✅ Done generating dataset for all A-Z!")
        cv2.destroyAllWindows()


# Main
image = cv2.imread(image_path)
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_crop)

print(f"[START] Draw a rectangle around each hand sign in order (A → Z)")
print(f"[→] Start with letter '{labels[current_label_index]}'")

while True:
    temp = clone.copy()
    if cropping and len(ref_point) == 1:
        cv2.rectangle(temp, ref_point[0], (cv2.getWindowImageRect("image")[
                      2], cv2.getWindowImageRect("image")[3]), (0, 255, 0), 2)

    cv2.imshow("image", temp)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

cv2.destroyAllWindows()
