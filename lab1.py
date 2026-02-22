import pandas as pd
input_file = pd.read_csv('50_Startups.csv')
data = pd.get_dummies(input_file, columns = ['State'], drop_first = True)
X = data.drop('Profit', axis = 1)
y = data['Profit']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error, r2_score
y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R^2 Score: {r2}")
import seaborn as sns
import matplotlib.pyplot as plt

sns.regplot(x = 'R&D Spend', y = 'Profit', data = input_file)
plt.title('R&D Spend vs Profit')
plt.show()