import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ---------------- Load Data ---------------- #
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'], dtype=np.float32)
labels = np.asarray(data_dict['labels'])

# ---------------- Data Augmentation ---------------- #
# Add Gaussian noise for robustness
noise = np.random.normal(0, 0.005, data.shape)
augmented_data = np.concatenate((data, data + noise), axis=0)
augmented_labels = np.concatenate((labels, labels), axis=0)

# ---------------- Train/Test Split ---------------- #
x_train, x_test, y_train, y_test = train_test_split(
    augmented_data, augmented_labels, test_size=0.2, stratify=augmented_labels, random_state=42
)

# ---------------- Model Training ---------------- #
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)
model.fit(x_train, y_train)

# ---------------- Evaluation ---------------- #
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)
print(f'Test Accuracy: {score * 100:.2f}%')

cv_scores = cross_val_score(model, augmented_data, augmented_labels, cv=5)
print(
    f'Cross-validation Accuracy: {np.mean(cv_scores) * 100:.2f}% ± {np.std(cv_scores) * 100:.2f}%')

cm = confusion_matrix(y_test, y_predict)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=np.unique(labels))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# ---------------- Save Model ---------------- #
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("✅ Improved model saved to model.p")
