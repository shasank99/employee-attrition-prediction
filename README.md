# Employee Attrition Prediction

A machine learning web application that predicts whether an employee is likely to leave an organization based on employee-related factors.
## 📸 Application Screenshot

![Employee Attrition Prediction](project-screenshot.pn

## Project Overview

Employee attrition can affect productivity, recruitment costs, and workforce planning. This project uses machine learning to identify patterns associated with employee attrition.

The application allows users to enter employee information through a web interface and receive an attrition prediction.

## Features

- Employee attrition prediction
- Data preprocessing
- Random Forest classification
- Flask web application
- Interactive prediction form
- Model evaluation using accuracy and classification metrics
- Saved trained machine learning model

## Dataset

The project uses the IBM HR Analytics Employee Attrition dataset.

The dataset contains:

- 1,470 employee records
- 35 features

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- Joblib
- HTML
- CSS
- Git
- GitHub

## Machine Learning

The current model uses:

**Random Forest Classifier**

The model was evaluated using a held-out test set.

Test accuracy:

**80.61%**

The classification report is also used to evaluate precision, recall, and F1-score.

## Project Structure

```text
employee-attrition-prediction/
│
├── data/
├── models/
├── static/
├── templates/
├── app.py
├── train_model.py
├── train_real_model.py
├── requirements.txt
├── README.md
└── .gitignore
