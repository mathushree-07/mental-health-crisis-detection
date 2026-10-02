import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load cleaned dataset
df = pd.read_csv("data/cleaned_dataset.csv")

# Create charts folder if it doesn't exist
os.makedirs("static/charts", exist_ok=True)

# 1. Depression Distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Depression")
plt.title("Depression Distribution")
plt.xlabel("Depression")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("static/charts/depression_distribution.png")
plt.close()


# 2. Academic Pressure vs Depression
plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="Academic Pressure",
    hue="Depression"
)
plt.title("Academic Pressure vs Depression")
plt.xlabel("Academic Pressure")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("static/charts/academic_pressure.png")
plt.close()


# 3. Suicidal Thoughts vs Depression
plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="Have you ever had suicidal thoughts ?",
    hue="Depression"
)
plt.title("Suicidal Thoughts vs Depression")
plt.xlabel("Suicidal Thoughts (0 = No, 1 = Yes)")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("static/charts/suicidal_thoughts.png")
plt.close()


# 4. Sleep Duration vs Depression
plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="Sleep Duration",
    hue="Depression"
)
plt.title("Sleep Duration vs Depression")
plt.xlabel("Sleep Duration")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("static/charts/sleep_duration.png")
plt.close()


print("All EDA charts generated successfully!")