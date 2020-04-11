from sklearn.linear_model import LinearRegression
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