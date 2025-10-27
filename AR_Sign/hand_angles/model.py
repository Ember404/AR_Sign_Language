import tensorflow as tf
import pandas as pd
from rich import columns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#add hand angle
import numpy as np
import pandas as pd

# Load your annotations CSV
df_all = pd.read_csv("../data.csv")

# Compute components
dx_AB = df_all["x9"] - df_all["x0"]
dy_AB = df_all["y9"] - df_all["y0"]
dx_BC = 0
dy_BC = df_all["y9"] - df_all["y0"]

# Compute dot product and magnitudes
dot = dy_AB * dy_BC  # (since dx_BC = 0)
mag_AB = np.sqrt(dx_AB**2 + dy_AB**2)
mag_BC = np.abs(dy_BC)

# Avoid division by zero
epsilon = 1e-8
cos_theta = dot / (mag_AB * mag_BC + epsilon)
cos_theta = np.clip(cos_theta, -1, 1)  # keep within valid range

# Compute angle in degrees
df_all["angle_hand"] = np.arccos(cos_theta)

df = pd.concat([df_all.iloc[:, :1], df_all.iloc[:, -16:]], axis=1)
print(df.head())
X = df.drop("letter", axis=1).values.astype("float32")  #drop labels

y = df["letter"].values

# Encode labels (if they are strings)
encoder = LabelEncoder()
y = encoder.fit_transform(y)



# Example: your DataFrame has columns like 'x0', 'y0', ..., 'x9', 'y9'
# df = pd.read_csv("annotations.csv")



print(X.shape)


# Split train/val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


num_classes = len(set(y))  # number of gesture classes

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(16,)),     # 15 angles
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

