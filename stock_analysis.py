import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# List of tickers for the companies you want to compare
tickers = ['AAPL', 'MSFT', 'TSLA', 'AMZN', 'TTWO', 'SONY', 'RL', 'META', 'NVDA', 'ARM']

# Download stock data for all the tickers
data = yf.download(tickers, start='2020-01-01', end='2024-08-20')['Close']

# Drop any rows with NaN values
data = data.dropna()

# Normalize the data to start from the same base (for relative comparison)
normalized_data = data / data.iloc[0]

# Create a figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(14, 14))

# Plotting the absolute closing prices
axs[0].set_title('Absolute Stock Closing Prices (2020-2024)')
for ticker in tickers:
    axs[0].plot(data.index, data[ticker], label=ticker)
axs[0].set_xlabel('Date')
axs[0].set_ylabel('Closing Price (USD)')
axs[0].grid(True)
axs[0].legend()

# Plotting the normalized closing prices for relative performance
axs[1].set_title('Relative Stock Closing Prices (2020-2024)')
for ticker in tickers:
    axs[1].plot(normalized_data.index, normalized_data[ticker], label=ticker)
axs[1].set_xlabel('Date')
axs[1].set_ylabel('Normalized Closing Price')
axs[1].grid(True)
axs[1].legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

# Function to predict future prices and create a visualized table
def predict_future_prices(data, months=6):
    future_predictions = {}
    future_dates = pd.date_range(data.index[-1], periods=months * 21, freq='B')  # Corrected to use pd.date_range
    
    plt.figure(figsize=(14, 7))
    for ticker in tickers:
        # Preparing the data
        X = np.arange(len(data[ticker])).reshape(-1, 1)
        y = data[ticker].values
        
        # Fitting the linear regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predicting future prices
        future_X = np.arange(len(data[ticker]) + months * 21).reshape(-1, 1)  # Approx. 21 trading days per month
        future_prices = model.predict(future_X)
        
        future_predictions[ticker] = future_prices[-months * 21:]  # Get the last n months of predictions
        
        # Plotting the predictions
        plt.plot(future_dates, future_prices[-months * 21:], label=f'Predicted {ticker} Prices', linestyle='--')
    
    plt.title('Predicted Stock Prices for the Next 6 Months')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Predict the prices for the next 6 months
predict_future_prices(data, months=6)
