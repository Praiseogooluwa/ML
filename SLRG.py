import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport

# Load the dataset
data= pd.read_csv('5G_energy_consumption_dataset.csv')

print(data.info())
print(data.describe())

profile = ProfileReport(data, title='5G-Energy consumption Profiling Report By OG')
profile.to_file('5G-Energy consumption_report_OG_Graphix.html')

print(data.isnull().sum())

print(data.duplicated().sum())

data.boxplot()
plt.show()

categorical_features = data.select_dtypes(include=['object']).columns
print(categorical_features)

from sklearn.model_selection import train_test_split

X = data[['load', 'ESMODE', 'TXpower']]  # Assuming 'energy_consumption' is the target variable
y = data['Energy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

