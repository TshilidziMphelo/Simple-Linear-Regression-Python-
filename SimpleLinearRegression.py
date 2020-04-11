import pandas as pd

# Load the data into a pandas DataFrame
df = pd.read_csv('score.csv')

# Explore the data
print(df.head())
print(df.describe())