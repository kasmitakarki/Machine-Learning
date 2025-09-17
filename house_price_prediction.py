#Task 1 data exploration
#Load and inspect dataset

# Import necessary libraries

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the dataset

data = pd.read_csv('house_prices.csv')

# Inspect the structure of the dataset

print(data.head())  # Show the first few rows
print(data.info())  # Show data types and check for missing values
print(data.describe())  # Summary statistics

# Scatter plot to visualize the relationship between size and price


plt.scatter(data['size'], data['price'], color='blue')
plt.title('House Size vs Price')
plt.xlabel('Size (square feet)')
plt.ylabel('Price (thousands of dollars)')
plt.grid(True)
plt.savefig('scatterplot.png') #saves the plot
plt.show()

#Task 2 Model Building


# Split the data into features (X) and target (y)
X = data[['size']]  # Feature: house size
y = data['price']   # Target: house price

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict house prices on the test set
y_pred = model.predict(X_test)

#Task 3 Model evaluation

# Calculate the Mean Squared Error (MSE)


mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# Task 4 Reporting your findings

#Plot the regression line along with the data points
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.title('Linear Regression: House Size vs Price')
plt.xlabel('Size (square feet)')
plt.ylabel('Price (thousands of dollars)')
plt.legend()
plt.show()

