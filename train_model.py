# ================================
# Iris Flower Classification using MLP
# Complete ML Pipeline Script
# ================================

# 1️⃣ Import Libraries

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import joblib

sns.set_style("whitegrid")


# =================================
# 2️⃣ Load Dataset
# =================================

print("\nLoading Dataset...\n")

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

print("Features:", feature_names)
print("Classes:", target_names)


# =================================
# 3️⃣ Convert to DataFrame
# =================================

df = pd.DataFrame(X, columns=feature_names)
df["target"] = y
df["species"] = df["target"].apply(lambda x: target_names[x])

print("\nDataset Preview:\n")
print(df.head())


# =================================
# 4️⃣ Basic EDA
# =================================

print("\nDataset Shape:", df.shape)

print("\nDataset Statistics:\n")
print(df.describe())


# Class Distribution Plot

plt.figure(figsize=(6,4))
sns.countplot(x="species", data=df)
plt.title("Class Distribution")
plt.tight_layout()
plt.show()


# Pairplot

sns.pairplot(df, hue="species")
plt.show()


# =================================
# 5️⃣ Train Test Split
# =================================

print("\nSplitting Dataset...\n")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)


# =================================
# 6️⃣ Feature Scaling
# =================================

print("\nScaling Features...\n")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =================================
# 7️⃣ Train Model
# =================================

print("\nTraining MLP Model...\n")

model = MLPClassifier(
    hidden_layer_sizes=(10,),
    max_iter=3000,
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed")


# =================================
# 8️⃣ Predictions
# =================================

print("\nMaking Predictions...\n")

y_pred = model.predict(X_test)


# =================================
# 9️⃣ Accuracy
# =================================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy,4))


# =================================
# 🔟 Confusion Matrix
# =================================

cm = confusion_matrix(y_test, y_pred)

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

plt.tight_layout()
plt.show()


# =================================
# 11️⃣ Classification Report
# =================================

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred, target_names=target_names))


# =================================
# 12️⃣ Cross Validation
# =================================

print("\nRunning Cross Validation...\n")

scaler_cv = StandardScaler()
X_scaled = scaler_cv.fit_transform(X)

model_cv = MLPClassifier(
    hidden_layer_sizes=(10,),
    max_iter=3000,
    random_state=42
)

scores = cross_val_score(model_cv, X_scaled, y, cv=5)

print("Cross Validation Scores:", scores)
print("Average Accuracy:", scores.mean())


# =================================
# 13️⃣ Save Model
# =================================

print("\nSaving Model...\n")

joblib.dump(model, "mlp_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully")


# =================================
# 14️⃣ Load Model
# =================================

print("\nLoading Saved Model...\n")

loaded_model = joblib.load("mlp_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")


# =================================
# 15️⃣ Test Prediction
# =================================

sample = np.array([[5.1, 3.5, 1.4, 0.2]])

sample_scaled = loaded_scaler.transform(sample)

prediction = loaded_model.predict(sample_scaled)

print("\nSample Prediction:")

print("Input:", sample)
print("Predicted Flower Species:", target_names[prediction][0])


print("\nPipeline Completed Successfully 🚀")
