import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
# test test
# from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import (StandardScaler,
                                   PolynomialFeatures)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# from sklearn.model_selection import train_test_split
# test test

df = pd.read_csv('data/co2_emissions.csv')

pd.set_option('display.max_columns', None)  # show all the columns
print(df.describe())
print(df.head())
print(df.dtypes)
# print(df.isnull().sum()) non vi sono elementi nulli

# engine_size,cylinders,fuel_consumption_hwy,fuel_consumption_comb(l/100km),fuel_consumption_comb(mpg),co2_emissions

# list(df.columns)
# df.drop(labels=['car name'], axis=1, inplace=True)
plt.figure(figsize=[14, 6])
sns.barplot(x=df['fuel_type'], y=df['fuel_consumption_comb(l/100km)'])
plt.title('Consumption Gallon by Fuel Type')
# plt.show()
# print(df.head())

# Exploratory data Analysis visualization and analysis
# plt.figure(figsize=(10, 8))
# sns.histplot(df.mpg)
# plt.show()

# Correlation
f, ax = plt.subplots(figsize=[14, 8])
sns.heatmap(df.corr(), annot=True, fmt=".2f")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# sns.pairplot(df, diag_kind='kde')
# plt.show()


# print(df.corr('spearman'))


# df['acceleration_power_ratio'] = df['acceleration'] / df['horsepower']




df.drop(labels=['make', 'model', 'vehicle_class', 'transmission', 'fuel_type', 'fuel_consumption_comb(mpg)'], axis=1,
        inplace=True)

#standardizzazione
scaler = StandardScaler()
scaled_array = scaler.fit_transform(df)
scaled_dataframe = pd.DataFrame( scaled_array, columns = df.columns )
plt.figure(figsize = (15,4))
sns.boxplot(data = scaled_dataframe, orient = "h")
plt.show()

print(scaled_dataframe.describe())

y = df['co2_emissions']
df.drop('co2_emissions', axis=1, inplace=True)
# print(df.head())


kmeans_model = KMeans(n_clusters = 3) #3 da elbow
kmeans_model.fit(scaled_dataframe)
centroids = kmeans_model.cluster_centers_
print(centroids)
print(kmeans_model.cluster_centers_.shape)

