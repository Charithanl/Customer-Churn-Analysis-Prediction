import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# give you file path here
dataset = pd.read_csv('Telco-Customer-Churn.csv')

dataset.head()

# Understanding the Dataset
print(dataset.isnull().sum())
print(dataset.describe())

