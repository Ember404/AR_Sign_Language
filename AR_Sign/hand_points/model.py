import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load your annotations CSV
df = pd.read_csv("../data.csv",usecols=range(43))
X = df.drop("letter", axis=1).values.astype("float32")  #drop labels
y = df["letter"].values

# Encode labels (if they are strings)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split train/val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


num_classes = len(set(y))  # number of gesture classes

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(42,)),     # 21 keypoints × (x, y)
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32
)

loss, acc = model.evaluate(X_val, y_val)
print(f"Validation accuracy: {acc:.3f}")

import numpy as np

y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_val, y_pred_classes)
print(cm)

class_names = encoder.classes_


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Mistake Matrix)")
plt.show()




