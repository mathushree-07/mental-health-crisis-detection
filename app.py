from flask import Flask, render_template, request
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load("depression_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/assessment")
def assessment():
    return render_template("assessment.html")

@app.route("/about")
def about():
    return render_template("about.html")
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from the assessment form
    age = request.form.get("age")
    gender = request.form.get("gender")
    city = request.form.get("city")
    profession = request.form.get("profession")
    degree = request.form.get("degree")

    academic_pressure = request.form.get("academic_pressure")
    work_pressure = request.form.get("work_pressure")

    cgpa = request.form.get("cgpa")

    study_satisfaction = request.form.get("study_satisfaction")
    job_satisfaction = request.form.get("job_satisfaction")

    sleep_duration = request.form.get("sleep_duration")
    dietary_habits = request.form.get("dietary_habits")

    work_study_hours = request.form.get("work_study_hours")
    financial_stress = request.form.get("financial_stress")

    suicidal_thoughts = request.form.get("suicidal_thoughts")
    family_history = request.form.get("family_history")


    # ------------------------------------------------
    # Convert numerical values
    # ------------------------------------------------

    age = float(age)
    academic_pressure = float(academic_pressure)
    work_pressure = float(work_pressure)
    cgpa = float(cgpa)
    study_satisfaction = float(study_satisfaction)
    job_satisfaction = float(job_satisfaction)
    work_study_hours = float(work_study_hours)
    financial_stress = float(financial_stress)


    # ------------------------------------------------
    # Convert Yes / No values
    # ------------------------------------------------

    suicidal_thoughts = 1 if suicidal_thoughts == "Yes" else 0
    family_history = 1 if family_history == "Yes" else 0


    # ------------------------------------------------
    # Convert Sleep Duration
    # ------------------------------------------------

    sleep_mapping = {
        "Less than 5 hours": 1,
        "5-6 hours": 2,
        "7-8 hours": 3,
        "More than 8 hours": 4
    }

    sleep_duration = sleep_mapping.get(sleep_duration)


    # ------------------------------------------------
    # Convert Dietary Habits
    # ------------------------------------------------

    diet_mapping = {
        "Unhealthy": 1,
        "Moderate": 2,
        "Healthy": 3
    }

    dietary_habits = diet_mapping.get(dietary_habits)


    # ------------------------------------------------
    # Create input DataFrame
    # ------------------------------------------------

    input_data = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "City": city,
        "Profession": profession,
        "Academic Pressure": academic_pressure,
        "Work Pressure": work_pressure,
        "CGPA": cgpa,
        "Study Satisfaction": study_satisfaction,
        "Job Satisfaction": job_satisfaction,
        "Sleep Duration": sleep_duration,
        "Dietary Habits": dietary_habits,
        "Degree": degree,
        "Have you ever had suicidal thoughts ?": suicidal_thoughts,
        "Work/Study Hours": work_study_hours,
        "Financial Stress": financial_stress,
        "Family History of Mental Illness": family_history
    }])


    # ------------------------------------------------
    # Predict depression probability
    # ------------------------------------------------

    probability = model.predict_proba(input_data)[0][1]

    probability_percentage = round(probability * 100, 2)


    # ------------------------------------------------
    # Determine risk category
    # ------------------------------------------------

    if probability <= 0.30:
        risk_level = "Low Risk"

    elif probability <= 0.70:
        risk_level = "Moderate Risk"

    else:
        risk_level = "High Risk"


    # ------------------------------------------------
    # Display result
    # ------------------------------------------------

    return render_template(
        "result.html",
        probability=probability_percentage,
        risk_level=risk_level
    )


if __name__ == "__main__":
    app.run(debug=True)