# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 07-09-2026



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```
# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')


# Step 2: Read the CSV file into a DataFrame
data = pd.read_csv('/content/Walmart_Sales.csv')

# Display column names
print("Column names in the dataset:")
print(data.columns)


# Convert 'Date' column to datetime
# Date format in the dataset is DD-MM-YYYY
data['Date'] = pd.to_datetime(
    data['Date'],
    format='%d-%m-%Y'
)


# Aggregate Weekly Sales of all stores by Date
walmart_sales = data.groupby('Date')['Weekly_Sales'].sum()

# Sort by Date
walmart_sales = walmart_sales.sort_index()


# Step 3: Perform Augmented Dickey-Fuller test for stationarity
result = adfuller(walmart_sales.dropna())

print('\nADF Statistic:', result[0])
print('p-value:', result[1])

if result[1] < 0.05:
    print("The time series is stationary.")
else:
    print("The time series is not stationary.")


# Step 4: Split the data into training and testing sets
# 80% training and 20% testing
train_size = int(len(walmart_sales) * 0.8)

train_data = walmart_sales.iloc[:train_size]
test_data = walmart_sales.iloc[train_size:]


# Step 5: Fit an AutoRegressive (AR) model with 13 lags
model = AutoReg(
    train_data,
    lags=13,
    old_names=False
)

ar_model_fit = model.fit()


# Step 6: Plot ACF
plt.figure(figsize=(10, 5))

plot_acf(
    train_data,
    lags=40
)

plt.title('Autocorrelation Function (ACF)')
plt.show()


# Plot PACF
plt.figure(figsize=(10, 5))

plot_pacf(
    train_data,
    lags=40
)

plt.title('Partial Autocorrelation Function (PACF)')
plt.show()


# Step 7: Make predictions using the AR model
predictions = ar_model_fit.predict(
    start=len(train_data),
    end=len(train_data) + len(test_data) - 1,
    dynamic=False
)


# Step 8: Compare actual values with predicted values
plt.figure(figsize=(12, 6))

plt.plot(
    test_data.index,
    test_data.values,
    label='Actual Walmart Sales'
)

plt.plot(
    test_data.index,
    predictions.values,
    label='Predicted Walmart Sales',
    linestyle='dashed'
)

plt.title('Actual vs Predicted Walmart Weekly Sales')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')

plt.legend()
plt.grid(True)
plt.show()


# Step 9: Calculate Mean Squared Error
mse = mean_squared_error(
    test_data,
    predictions
)

print('\nMean Squared Error (MSE):', mse)
```
### OUTPUT:

GIVEN DATA
<img width="631" height="71" alt="image" src="https://github.com/user-attachments/assets/555ba786-21c2-4229-93be-e81975a7f96b" />


PACF - ACF

<img width="962" height="630" alt="image" src="https://github.com/user-attachments/assets/19a45266-0583-440c-a29c-6dec19f84020" />

PREDICTION
<img width="1052" height="463" alt="image" src="https://github.com/user-attachments/assets/b7863c46-ef35-4714-bf0b-b36d1409462d" />



### RESULT:
Thus we have successfully implemented the auto regression function using python.
