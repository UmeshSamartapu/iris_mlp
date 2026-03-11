Below is a **similar Jupyter Notebook style project**, but using a **Convolutional Neural Network (CNN)** instead of MLP.

Important note:
The Iris dataset has **4 tabular features**, not images.
So we reshape the data into a **1D signal** and use a **1D CNN (Conv1D)**.

This is a **good beginner CNN example for tabular data**.

You can paste each section into **separate Jupyter notebook cells**.

---

# 1️⃣ Import Libraries

```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.utils import to_categorical

import joblib

sns.set_style("whitegrid")
```

---

# 2️⃣ Load Dataset

```python
iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

print("Features:", feature_names)
print("Classes:", target_names)
```

---

# 3️⃣ Convert to DataFrame

```python
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y
df["species"] = df["target"].apply(lambda x: target_names[x])

df.head()
```

---

# 4️⃣ Basic EDA

```python
print("Dataset Shape:", df.shape)

df.describe()
```

---

# Class Distribution

```python
plt.figure(figsize=(6,4))

sns.countplot(x="species", data=df)

plt.title("Class Distribution")

plt.show()
```

---

# Pairplot

```python
sns.pairplot(df, hue="species")
plt.show()
```

---

# 5️⃣ Train Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)
```

---

# 6️⃣ Feature Scaling

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

# 7️⃣ Reshape Data for CNN

CNN expects **3D input**.

Shape format:

```
(samples, timesteps, features)
```

```python
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print("CNN Input Shape:", X_train.shape)
```

Example output

```
(120, 4, 1)
```

---

# 8️⃣ One-Hot Encode Labels

```python
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
```

---

# 9️⃣ Build CNN Model

Architecture:

```
Conv1D
MaxPooling
Flatten
Dense
Output Layer
```

```python
model = Sequential()

model.add(Conv1D(
    filters=32,
    kernel_size=2,
    activation="relu",
    input_shape=(4,1)
))

model.add(MaxPooling1D(pool_size=1))

model.add(Flatten())

model.add(Dense(16, activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(3, activation="softmax"))
```

---

# 🔟 Compile Model

```python
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
```

---

# 11️⃣ Train Model

```python
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=8,
    validation_split=0.2,
    verbose=1
)
```

---

# 12️⃣ Evaluate Model

```python
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)
```

Expected accuracy

```
0.95 – 1.00
```

---

# 13️⃣ Predictions

```python
y_pred_probs = model.predict(X_test)

y_pred = np.argmax(y_pred_probs, axis=1)

y_true = np.argmax(y_test, axis=1)
```

---

# 14️⃣ Confusion Matrix

```python
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()
```

---

# 15️⃣ Classification Report

```python
print(classification_report(y_true, y_pred, target_names=target_names))
```

---

# 16️⃣ Save Model

CNN models are saved as **.h5**

```python
model.save("iris_cnn_model.h5")

joblib.dump(scaler, "scaler.pkl")

print("CNN model saved successfully")
```

---

# 17️⃣ Load Model

```python
from tensorflow.keras.models import load_model

loaded_model = load_model("iris_cnn_model.h5")

loaded_scaler = joblib.load("scaler.pkl")
```

---

# 18️⃣ Test Prediction

```python
sample = np.array([[5.1, 3.5, 1.4, 0.2]])

sample_scaled = loaded_scaler.transform(sample)

sample_scaled = sample_scaled.reshape(1,4,1)

prediction = loaded_model.predict(sample_scaled)

predicted_class = np.argmax(prediction)

print("Predicted Flower Species:", target_names[predicted_class])
```

Example output

```
Predicted Flower Species: Setosa
```

---

# ✔ Final Project Structure

```
iris_cnn_project
│
├── iris_cnn.ipynb
├── iris_cnn_model.h5
├── scaler.pkl
├── app.py
├── requirements.txt
```

---

# requirements.txt

```
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
fastapi
uvicorn
joblib
```

Install

```
pip install -r requirements.txt
```

---

# ⭐ CNN Architecture Used

```
Input (4 features)
      ↓
Conv1D (32 filters)
      ↓
MaxPooling
      ↓
Flatten
      ↓
Dense (16)
      ↓
Dropout
      ↓
Dense (3 Softmax)
```

---

# Final Deep Learning Pipeline

```
Problem Definition
        ↓
Data Collection
        ↓
EDA
        ↓
Preprocessing
        ↓
Train/Test Split
        ↓
CNN Model Training
        ↓
Evaluation
        ↓
Confusion Matrix
        ↓
Model Saving
        ↓
Deployment
```

---

💡 Since you are learning **ML + DL pipelines**, the **next powerful CNN beginner projects** are:

1️⃣ **MNIST Digit Recognition (CNN)**
2️⃣ **Plant Disease Detection (CNN)**
3️⃣ **Chest X-ray Pneumonia Detection (CNN)**
4️⃣ **Traffic Sign Classification (CNN)**

Those are **true CNN projects used in portfolios**.

---

If you want, I can also show you the **BEST CNN beginner project (MNIST)** that teaches:

* CNN architecture
* Feature maps
* pooling layers
* training curves
* **99% accuracy**

It is **the most recommended first CNN project in deep learning.**
