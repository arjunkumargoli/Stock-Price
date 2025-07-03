import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Download historical data
df = yf.download('AAPL', start='2010-01-01', end='2024-12-31')
df = df[['Close']]
df.dropna(inplace=True)

# Step 2: Prepare data
future_days = 30
df['Prediction'] = df[['Close']].shift(-future_days)

X = df[['Close']][:-future_days]
y = df['Prediction'][:-future_days]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Make predictions and calculate error
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Step 5: Predict future prices
X_future = df[['Close']][-future_days:]
future_predictions = model.predict(X_future)

# Step 6: Visualize results
plt.figure(figsize=(12,6))
plt.plot(df.index[-100:], df['Close'][-100:], label='Actual Prices')
plt.plot(df.index[-future_days:], future_predictions, label='Predicted Prices', linestyle='dashed')
plt.title('Stock Price Prediction (AAPL)')
plt.xlabel('Date')
plt.ylabel('Close Price USD')
plt.legend()
plt.grid(True)
plt.show()
