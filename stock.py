import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
ticker_symbol = "Apple.inc"

data = yf.download(ticker_symbol, start="2020-01-01", end="2023-01-01")

print("Stock:", ticker_symbol)

# After prediction

# Step 1: Download stock data
data = yf.download("AAPL", start="2020-01-01", end="2023-01-01")

# Step 2: Use only closing price
data = data[['Close']]

# Step 3: Create target (next day price)
data['Target'] = data['Close'].shift(-1)

# Remove last row (it has NaN target)
data = data.dropna()

# Step 4: Features (X) and Labels (y)
X = data[['Close']]
y = data['Target']

# Step 5: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 6: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Predict
predictions = model.predict(X_test)

# Step 8: Show results
print("\nPredictions for:", ticker_symbol)
print("Predicted Prices:", predictions[:5])
print("Predicted Prices:", predictions[:5])
print("Actual Prices:", y_test.values[:5])
