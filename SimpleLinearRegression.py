import pandas as pd

# Load the data into a pandas DataFrame
df = pd.read_csv('score.csv')

# Explore the data
print(df.head())
print(df.describe())

# Clean and prepare the data
df = df.dropna()

# Select the features
X = df.iloc[:, :-1]

# Select the target variable
y = df.iloc[:, -1]