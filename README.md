# Hotel Cancellation Prediction

## 📌 Overview
This project predicts hotel booking cancellations using machine learning models.  
It includes data preprocessing, model training, and an interactive [Streamlit](https://streamlit.io) app for real-time predictions.

## 🚀 Features
- Data cleaning and preprocessing of hotel booking dataset.
- Machine learning model trained with scikit-learn and saved using joblib.
- Streamlit web app for interactive predictions.
- Easy deployment and reproducibility.

## 📂 Project Structure
hotel-cancellation-prediction/
│── app.py                # Streamlit app
│── model.pkl             # Trained ML model (joblib)
│── requirements.txt      # Dependencies
│── README.md             # Project documentation
│── data/                 # Dataset (if included)
│── notebooks/            # Jupyter notebooks for exploration


## ⚙️ Installation & Usage
Clone the repository and install dependencies:

git clone https://github.com/your-username/hotel-cancellation-prediction.git
cd hotel-cancellation-prediction
pip install -r requirements.txt

## Run App 
streamlit run app.py

## Dataset
The dataset used is the Hotel Booking Demand Dataset.
It contains booking details such as lead time, arrival date, number of guests, and cancellation status.

## Model
Preprocessing: Handling missing values, encoding categorical variables.
Model: Classification algorithm (e.g., Random Forest, Logistic Regression).
Evaluation: Accuracy, precision, recall, F1-score.
