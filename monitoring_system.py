import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    "temperature": [25, 28, 30, 32, 35],
    "humidity": [60, 65, 70, 72, 75],
    "performance": [80, 76, 72, 68, 63]
}

df = pd.DataFrame(data)

X = df[["temperature", "humidity"]]
y = df["performance"]

model = LinearRegression()
model.fit(X, y)

new_data = [[31, 71]]
prediction = model.predict(new_data)

print("Monitoring Data:")
print(df)

print("\nPredicted Performance:", prediction[0])