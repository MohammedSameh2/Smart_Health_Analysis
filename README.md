# Smart Health Analysis

A machine learning project that predicts potential health conditions using common blood biomarkers through a simple Flask web app.

---

## 🔍 What It Does

* Lets users input health markers through a web form.
* Uses trained machine learning models to classify potential conditions.
* Supports multiple health categories: **Fit**, **Anemia**, **Hypertension**, **Diabetes**, and **High Cholesterol**.

---

## 📊 Biomarkers Used

* HbA1C
* Systolic Blood Pressure
* Diastolic Blood Pressure
* LDL Cholesterol
* HDL Cholesterol
* Triglycerides

---

## 📂 Project Files

```
Smart_Health_Analysis/
├── app.py                       # Main Flask application
├── for_deploy.py                # Alternative entrypoint for deployment
├── templates/                   # Web app HTML templates
├── health_markers_dataset.csv   # Training dataset
├── health-markers-dataset.ipynb # Data exploration and model training notebook
├── Voting_health_model.pkl      # Trained ensemble model
├── health_markers_dataset_model.pkl # Baseline model
└── .venv/                       # Virtual environment (local only)
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/MohammedSameh2/Smart_Health_Analysis.git
cd Smart_Health_Analysis
```

### 2. Create and Activate a Virtual Environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

If `requirements.txt` is available:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install flask scikit-learn pandas numpy joblib matplotlib
```

### 4. Run the Application

```bash
python app.py
```

Open your browser at **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**.

---

## 🧠 The Models

* **Baseline model** (`health_markers_dataset_model.pkl`) trained directly on the dataset.
* **Ensemble model** (`Voting_health_model.pkl`) that combines multiple algorithms for better accuracy.
* Training workflow and experiments are detailed in the Jupyter notebook.

---

## 📘 Example Usage

```python
import joblib
import numpy as np

model = joblib.load('Voting_health_model.pkl')
example = np.array([[6.2, 130, 82, 160, 45, 180]])
print(model.predict(example))
```

---

## 🙌 Credits

Developed by **Mohamed Sameh** using Python, Flask, and scikit-learn.
