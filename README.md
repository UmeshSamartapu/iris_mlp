# 🌸 Iris Flower Classification using MLP (Machine Learning)

A complete **Machine Learning pipeline project** that classifies iris flowers into three species using a **Multilayer Perceptron (MLP) neural network** built with **scikit-learn**.

This project demonstrates the **end-to-end ML workflow** including:

* Data loading
* Exploratory Data Analysis (EDA)
* Data preprocessing
* Model training
* Model evaluation
* Cross-validation
* Model saving
* API deployment using FastAPI

---

# 📌 Project Objective

The goal of this project is to build a **machine learning model that predicts the species of an iris flower** based on four flower measurements:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

The model classifies the flower into one of three species:

* Setosa
* Versicolor
* Virginica

---

# 📊 Dataset

The dataset used is the **Iris Dataset**, a well-known dataset in machine learning.

Dataset contains:

* **150 samples**
* **4 input features**
* **3 output classes**

Features:

| Feature      | Description          |
| ------------ | -------------------- |
| Sepal Length | Length of sepal (cm) |
| Sepal Width  | Width of sepal (cm)  |
| Petal Length | Length of petal (cm) |
| Petal Width  | Width of petal (cm)  |

Target Classes:

| Class      | Label |
| ---------- | ----- |
| Setosa     | 0     |
| Versicolor | 1     |
| Virginica  | 2     |

---

# 🧠 Machine Learning Model

The model used in this project is:

**Multilayer Perceptron (MLPClassifier)**

Architecture:

```
Input Layer (4 features)
        ↓
Hidden Layer (10 neurons)
        ↓
Output Layer (3 classes)
```

Library Used:

* scikit-learn

---

# 🔬 Machine Learning Pipeline

The project follows a **standard ML workflow**:

```
Problem Definition
        ↓
Data Collection
        ↓
Exploratory Data Analysis (EDA)
        ↓
Data Preprocessing
        ↓
Train/Test Split
        ↓
Feature Scaling
        ↓
Model Training (MLP)
        ↓
Model Evaluation
        ↓
Cross Validation
        ↓
Model Saving
        ↓
API Deployment
```

---

# 📈 Exploratory Data Analysis (EDA)

The following visualizations are used:

* Class Distribution Plot
* Pairplot Visualization
* Dataset Statistics

These help understand:

* Feature relationships
* Data balance
* Class separability

---

# 📊 Model Evaluation

Evaluation metrics used:

* Accuracy
* Confusion Matrix
* Classification Report
* Cross Validation

Expected Performance:

```
Accuracy: ~96% – 100%
Cross Validation Accuracy: ~97%
```

---

# 💾 Model Saving

The trained model and scaler are saved using **Joblib**.

Saved files:

```
mlp_model.pkl
scaler.pkl
```

This allows the model to be reused without retraining.

---

# 🚀 API Deployment

A **FastAPI REST API** is included to serve the model.

API endpoint:

```
/predict
```

Example request:

```
http://127.0.0.1:8000/predict?sepal_length=5.1&sepal_width=3.5&petal_length=1.4&petal_width=0.2
```

Example response:

```
{
  "prediction": "Setosa"
}
```

Interactive API documentation:

```
http://127.0.0.1:8000/docs
```

---

# 📂 Project Structure

```
iris_mlp_project
│
├── train_model.py       # ML pipeline script
├── app.py               # FastAPI deployment
├── mlp_model.pkl        # Trained model
├── scaler.pkl           # Feature scaler
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

---

# ⚙️ Installation

Clone the repository:

```
git clone https://github.com/yourusername/iris-mlp-project.git
cd iris-mlp-project
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# ▶️ Run the ML Pipeline

Train the model:

```
python train_model.py
```

This will:

* Train the model
* Evaluate performance
* Save the model

---

# 🌐 Run the API

Start the FastAPI server:

```
uvicorn app:app --reload
```

Open browser:

```
http://127.0.0.1:8000/docs
```

---

# 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* FastAPI
* Uvicorn

---

# 📚 Learning Outcomes

This project demonstrates:

* Complete ML workflow
* Data preprocessing techniques
* Neural network classification
* Model evaluation techniques
* Cross validation
* Model deployment as API

---

# 📌 Future Improvements

Possible improvements:

* Use **Pipeline for preprocessing + model**
* Hyperparameter tuning using **GridSearchCV**
* Add **Docker containerization**
* Deploy API on **Render / AWS / GCP**
* Add **model explainability (SHAP)**

---

# 👨‍💻 Author

Developed as a **Machine Learning learning project** demonstrating a full ML pipeline and deployment.

---
"# iris_mlp" 
