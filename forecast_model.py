"""
forecast_model.py
Synthetic demo of ship maintenance forecasting using linear regression.
⚠️ NOTE: This uses fake data and is for demonstration purposes only.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --- Generate synthetic data ---
np.random.seed(42)
months = np.arange(1, 25)  # 24 months
maintenance_days = 20 + 0.5 * months + np.random.randint(-3, 3, size=len(months))

data = pd.DataFrame({
    'Month': months,
    'Maintenance_Days': maintenance_days
})

print("Sample data:")
print(data.head())

# --- Train simple regression model ---
X = data[['Month']]
y = data['Maintenance_Days']

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)

# --- Plot results ---
plt.figure(figsize=(8, 5))
plt.scatter(data['Month'], data['Maintenance_Days'], color='blue', label='Actual')
plt.plot(data['Month'], predictions, color='red', linewidth=2, label='Forecast')
plt.title('Synthetic Ship Maintenance Forecast')
plt.xlabel('Month')
plt.ylabel('Maintenance Days')
plt.legend()
plt.tight_layout()
plt.show()
