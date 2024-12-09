import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


titanic_data = pd.read_csv("titanic.csv")
print(titanic_data.isnull().sum())  
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
print(titanic_data.info())
print(titanic_data.describe())
sns.countplot(x='Sex', hue='Survived', data=titanic_data)
plt.title('Survival by Sex')
plt.show()
sns.countplot(x='Pclass', hue='Survived', data=titanic_data)
plt.title('Survival by Pclass')
plt.show()
sns.histplot(data=titanic_data, x='Age', kde=True)
plt.title('Age Distribution')
plt.show()


corr_matrix = titanic_data.corr()
sns.heatmap(corr_matrix, annot=True)
plt.title('Correlation Matrix')
plt.show()

sns.scatterplot(x='Age', y='Fare', hue='Survived', data=titanic_data)
plt.title('Age vs. Fare')
plt.show()


sns.countplot(x='Embarked', hue='Survived', data=titanic_data)
plt.title('Survival by Embarked Port')
plt.show()

