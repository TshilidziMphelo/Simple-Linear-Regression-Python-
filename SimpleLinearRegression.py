from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd

class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

# Load the data into a pandas DataFrame
df = pd.read_csv('score.csv')

# Explore the data
print(df.head())
print(df.describe())

# Cleaning the data
df = df.dropna()

# Select the features
X = df.iloc[:, :-1]

# Select the target variable
y = df.iloc[:, -1]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create an instance of the LinearRegressionModel class
model = LinearRegressionModel()

# Split the data into training and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = model.evaluate(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')