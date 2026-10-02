# Mental Health Crisis Detection

## AI-Based Student Depression Risk Screening System

A machine-learning-based web application designed to estimate depression risk among students using academic, lifestyle, financial, and mental-health-related factors.

> **Important:** This is an educational and early-screening prototype. It is not a medical diagnostic system.

---

## Project Objective

The objective of this project is to develop a machine-learning system that can estimate depression risk based on student-related factors.

The system accepts information through a web-based assessment form, processes the input, and uses a trained Logistic Regression model to estimate the probability of depression risk.

---

## Project Workflow

Student Assessment  
↓  
Input Preprocessing  
↓  
Logistic Regression Model  
↓  
Depression Probability  
↓  
Prototype Risk Category  
↓  
Screening Result

---

## Dataset

The project uses the Student Depression Dataset.

### Dataset Information

- Original records: 27,901
- Cleaned records: 27,868
- Target variable: `Depression`
- Problem type: Binary Classification

### Major Features

- Age
- Gender
- City
- Profession
- Academic Pressure
- Work Pressure
- CGPA
- Study Satisfaction
- Job Satisfaction
- Sleep Duration
- Dietary Habits
- Degree
- Suicidal Thoughts
- Work/Study Hours
- Financial Stress
- Family History of Mental Illness

---

## Data Preprocessing

The following preprocessing operations are performed:

1. The non-predictive `id` column is removed.
2. Records containing `?` in Financial Stress are removed.
3. Yes/No values are converted into binary values.
4. Sleep Duration is ordinally encoded.
5. Dietary Habits is ordinally encoded.
6. Categorical features are one-hot encoded.
7. Numerical features are standardized.
8. The cleaned dataset is saved as `cleaned_dataset.csv`.

---

## Machine Learning Model

The project uses **Logistic Regression** for binary classification.

The dataset is divided using an **80/20 stratified train-test split**.

The trained model estimates:

`P(Depression = 1)`

This probability is then mapped to a prototype screening category.

---

## Model Performance

Results obtained from the project's test set:

| Metric | Result |
|---|---:|
| Accuracy | 85.58% |
| Precision | 86.52% |
| Recall | 89.28% |
| F1 Score | 87.88% |

---

## Prototype Risk Categories

| Probability | Category |
|---|---|
| 0% – 30% | Low Risk |
| Above 30% – 70% | Moderate Risk |
| Above 70% | High Risk |

These categories are prototype screening categories created for this educational project and are **not clinically validated thresholds**.

---

## Technology Stack

### Frontend
- HTML
- CSS

### Backend
- Python
- Flask

### Machine Learning
- Scikit-learn
- Logistic Regression

### Data Processing
- Pandas
- NumPy

### Model Storage
- Joblib

### Visualization
- Matplotlib
- Seaborn

---

## Project Structure

```text
mental-health-crisis-detection/
│
├── data/
│   ├── Student Depression Dataset.csv
│   └── cleaned_dataset.csv
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│
├── templates/
│   ├── index.html
│   ├── assessment.html
│   ├── result.html
│   ├── about.html
│   └── dashboard.html
│
├── app.py
├── train_model.py
├── depression_model.pkl
├── requirements.txt
└── README.md