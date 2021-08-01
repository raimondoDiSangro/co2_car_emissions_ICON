import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
# test test
# from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import (StandardScaler,
                                   PolynomialFeatures)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier

from source.data_analysis import pair_plot
from source.data_analysis import correlation_heatmap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import pandas as pd

import tkinter as tk
from tkinter import messagebox
from source.classification_models import svr_model, kneigh_model, decision_tree_model

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

# from source.bayesian_network import bayesianNetwork
# from source.clustering import kMeansCluster


# engine_size,cylinders,fuel_consumption_city, fuel_consumption_hwy,fuel_consumption_comb(l/100km)

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
# plt.show()

# sns.pairplot(df, diag_kind='kde')
# plt.show()


# print(df.corr('spearman'))


# df['acceleration_power_ratio'] = df['acceleration'] / df['horsepower']


df.drop(labels=['make', 'model', 'vehicle_class', 'transmission', 'fuel_type', 'fuel_consumption_comb(mpg)'], axis=1,
        inplace=True)

y = df['co2_emissions']
df.drop('co2_emissions', axis=1, inplace=True)
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0)

model_pipe = Pipeline(steps=[('scaler', StandardScaler(),), ('lasso', LassoCV(),)])
# model = Pipeline('scaler', StandardScaler())
model_pipe.fit(X_train, y_train)
prediction = model_pipe.predict(X_test)
# print(prediction)

# print(X_test)
# print(prediction)
print("lassoCV train accuracy", model_pipe.score(X_train, y_train))
print("lassoCV test accuracy", model_pipe.score(X_test, y_test))

print("lassoCV mean absolute error:", mean_absolute_error(y_test, prediction))
print("lassoCV r2 score:", r2_score(y_test, prediction))
print("lassoCV mean squared error:", mean_squared_error(y_test, prediction))
print("lassoCV mean absolute error percentage",
      mean_absolute_percentage_error(y_test, prediction))  # 0.0 is the best

# todo
# n_features = 5
n_features = 3
#rng = np.random.RandomState(0)
regr = np.random.RandomState(0)
regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
regr.fit(X_train, y_train)
Pipeline(steps=[('standardscaler', StandardScaler()),
                ('svr', SVR(epsilon=0.2))])

print("svr train accuracy", regr.score(X_train, y_train))
print("svr train accuracy", regr.score(X_test, y_test))

prediction = regr.predict(X_test)
# print(prediction)

print("SVR mean absolute error:", mean_absolute_error(y_test, prediction))
print("SVR r2 score:", r2_score(y_test, prediction))
print("SVR mean squared error:", mean_squared_error(y_test, prediction))
print("SVR mean absolute error percentage",
      mean_absolute_percentage_error(y_test, prediction))  # 0.0 is the best

# cylinders,displacement,horsepower,weight,acceleration,model year,origin

# engine_size,cylinders,fuel_consumption_city, fuel_consumption_hwy,fuel_consumption_comb(l/100km)
# make,model,vehicle_class,engine_size,cylinders,transmission,fuel_type,fuel_consumption_city,fuel_consumption_hwy,fuel_consumption_comb(l/100km),fuel_consumption_comb(mpg),co2_emissions
# FIAT,500L,STATION WAGON - SMALL,1.4,4,M,X,9.3,7.1,8.3,34,194
# 2,4,5.5,6.7,42,181
# BMW,328d xDRIVE,COMPACT,2,4,A,D,7.6,5.5,6.7,42,181
# FIAT,500L,STATION WAGON - SMALL,1.4,4,M,X,9.3,7.1,8.3,34,194
# 1.4,4,9.3,7.1,8.3,34

#input_data = (1.4, 4, 9.3, 7.1, 8.3)  # exp. 194
# input_data = (6.5, 12, 25.2, 14.1, 16.1)  # ferrari 812 superfast  340 pred.
# input_data = (3.0, 6, 10, 8, 9)

# engine_size,cylinders,fuel_consumption_comb(l/100km)
# input_data = (1.4, 4, 8.3)  # exp. 194

input_data = [1.4, 4, 9.3, 7.1, 8.3]  # exp. 194

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model_pipe.predict(input_data_reshaped)
print(prediction)
prediction = regr.predict(input_data_reshaped)
print(prediction)

kMeans = KMeans(n_clusters=3)
kmodel = kMeans.fit(df)
# print(kmodel.predict(df))

distortions = []
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df)
    distortions.append(kmeanModel.inertia_)

plt.figure(figsize=(16, 8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
#plt.show()
