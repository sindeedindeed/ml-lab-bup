import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

input_file = pd.read_csv('50_Startups.csv')

print(f"Shape: {input_file.shape[0]} rows, {input_file.shape[1]} columns")
print(f"\nDtypes:\n{input_file.dtypes}")
print(f"\nSummary statistics:\n{input_file.describe()}")
print(f"\nState value counts:\n{input_file['State'].value_counts()}")

data = pd.get_dummies(input_file, columns=['State'], drop_first=True)
X = data.drop('Profit', axis=1)
y = data['Profit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE:  {mse}")
print(f"RMSE: {rmse}")
print(f"MAE:  {mae}")
print(f"R²:   {r2}")

print(f"Intercept: {regressor.intercept_:.4f}")
for name, coef in zip(X.columns, regressor.coef_):
    print(f"  {name}: {coef:.4f}")

metrics_table = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'MAE', 'R²'],
    'Value': [mse, rmse, mae, r2]
})
print(metrics_table.to_string(index=False))

sns.regplot(x='R&D Spend', y='Profit', data=input_file)
plt.title('R&D Spend vs Profit')
plt.show()

plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect prediction')
plt.xlabel('Actual Profit')
plt.ylabel('Predicted Profit')
plt.title('Actual vs Predicted Profit')
plt.legend()
plt.tight_layout()
plt.show()
