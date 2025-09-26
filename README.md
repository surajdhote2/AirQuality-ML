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
```bash
AirQuality-ML/
│── data/             # dataset
    ├── AirQualityUCI.xlsx              # main training dataset
    ├── Generalization_Dataset.xlsx     # test dataset
|
|── notebook/               # Jupyter notebook
│   ├── AirQuality_MLP.ipynb
|
│── src/                    # Python script
│   ├── classification_regression.py
|
│── outputs/            # plots, tables
│   │── figures
|   │── tables
|
|── .gitignore
│── LICENSE
│── requirements.txt # dependencies
│── README.md # project documentation


## 🚀 Run Locally
```bash
# Clone repo
git clone https://github.com/surajdhote2/AirQuality-ML.git
cd AirQuality-ML

# Install dependencies
pip install -r requirements.txt

# Run
python src/classification_regression.py
