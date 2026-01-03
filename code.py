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

# Analyzing Churn Distribution
print(dataset['Churn'].value_counts())
sns.countplot(x='Churn', data=dataset, palette='coolwarm')
plt.title('Churn Distribution')
plt.xlabel('Churn (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Data Preprocessing
dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'], errors='coerce')
dataset['TotalCharges'].fillna(dataset['TotalCharges'].median(), inplace=True)

from sklearn.preprocessing import LabelEncoder

