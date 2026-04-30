import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import os

df = pd.read_csv("sensor_data.csv")

print("\nIntelligent Monitoring System Report")
print(df)

print("\nDataset Summary:")
print(df.describe())

df["Risk_Level"] = df["Air_Quality_Index"].apply(
    lambda x: "High Risk" if x > 100 else "Medium Risk" if x > 70 else "Low Risk"
)

print("\nRisk Level Analysis:")
print(df[["Day", "Air_Quality_Index", "Risk_Level"]])

X = df[["Temperature", "Humidity", "Air_Quality_Index", "Machine_Load", "Energy_Usage"]]
y = df["Performance"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nModel Evaluation:")
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

new_data = pd.DataFrame({
    "Temperature": [34],
    "Humidity": [73],
    "Air_Quality_Index": [115],
    "Machine_Load": [76],
    "Energy_Usage": [180]
})

predicted_performance = model.predict(new_data)

print("\nPredicted Performance for New Sensor Data:")
print(predicted_performance[0])

if predicted_performance[0] < 70:
    print("Alert: System performance is low. Immediate action required.")
else:
    print("System performance is stable.")

os.makedirs("visualizations", exist_ok=True)

plt.figure()
plt.plot(df["Day"], df["Temperature"], marker="o", label="Temperature")
plt.plot(df["Day"], df["Humidity"], marker="o", label="Humidity")
plt.title("Temperature and Humidity Trend")
plt.xlabel("Day")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.savefig("visualizations/temperature_humidity_trend.png")

plt.figure()
plt.bar(df["Day"], df["Air_Quality_Index"])
plt.title("Air Quality Index Monitoring")
plt.xlabel("Day")
plt.ylabel("AQI")
plt.savefig("visualizations/air_quality_index.png")

plt.figure()
plt.plot(df["Day"], df["Performance"], marker="o")
plt.title("System Performance Trend")
plt.xlabel("Day")
plt.ylabel("Performance")
plt.grid()
plt.savefig("visualizations/performance_trend.png")

plt.figure()
plt.scatter(df["Energy_Usage"], df["Performance"])
plt.title("Energy Usage vs Performance")
plt.xlabel("Energy Usage")
plt.ylabel("Performance")
plt.savefig("visualizations/energy_vs_performance.png")

plt.show()