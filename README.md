# Air Quality Prediction (Classification & Regression)

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![Keras](https://img.shields.io/badge/Keras-TF-red)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📌 Overview
This project predicts **CO levels (classification)** and **NOx levels (regression)** using neural networks trained on the [UCI Air Quality dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality).

## ⚙️ Tech Stack
- Python, Pandas, NumPy, Scikit-learn
- Keras/TensorFlow
- Matplotlib, Seaborn

## 📂 Project Structure

AirQuality-ML/
│── data/ # dataset (not included)
│── src/ # source code
│ ├── preprocessing.py
│ ├── classification.py
│ ├── regression.py
│ ├── evaluation.py
│ └── train.py
│── models/ # saved models
│── outputs/ # plots, logs
│── requirements.txt # dependencies
│── README.md # project documentation


## 🚀 Run Locally
```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/AirQuality-ML.git
cd AirQuality-ML

# Install dependencies
pip install -r requirements.txt

# Run classification
python src/train.py --task classification

# Run regression
python src/train.py --task regression


📊 Results

Classification Accuracy: ~95%

Regression RMSE: ~30

🔮 Future Work

Hyperparameter tuning with GridSearch

Add Random Forest/XGBoost baselines

Deploy model using Flask/Streamlit