import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

print("Downloading USD/INR data...")

# Download forex data
data = yf.download("INR=X", start="2024-01-01", end="2026-06-04")

# Keep closing price
data = data[['Close']]

# Create tomorrow prediction column
data['Prediction'] = data['Close'].shift(-1)

# Remove empty row
data.dropna(inplace=True)

# Input and output
X = data[['Close']]
y = data['Prediction']

# Train ML model
model = LinearRegression()
model.fit(X, y)

# Latest available price
latest_price = data[['Close']].tail(1)

# Predict tomorrow
prediction = model.predict(latest_price)

print("\nPredicted USD to INR rate for tomorrow:")
print(round(prediction[0], 2))

# Plot graph
plt.figure(figsize=(10,5))
plt.plot(data.index, data['Close'])
plt.title("USD to INR Exchange Rate")
plt.xlabel("Date")
plt.ylabel("INR")
plt.grid(True)

plt.show()
