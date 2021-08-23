import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.cluster import KMeans

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
# test test
# from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

# engine_size,cylinders,fuel_consumption_city, fuel_consumption_hwy,fuel_consumption_comb(l/100km)

# from sklearn.model_selection import train_test_split
# test test

df = pd.read_csv('data/co2_emissions.csv')
#pd.set_option('display.max_columns', none)  # show all the columns

print(df.describe())
print(df.head())
print(df.dtypes)
print(df.duplicated().sum())
#df.drop_duplicates(inplace=true)
print(df.duplicated().sum())

# print(df.isnull().sum()) non vi sono elementi nulli

# engine_size,cylinders,fuel_consumption_hwy,fuel_consumption_comb(l/100km),fuel_consumption_comb(mpg),co2_emissions

# list(df.columns)
# df.drop(labels=['car name'], axis=1, inplace=true)
# plt.figure(figsize=[14, 6])
# sns.barplot(x=df['fuel_type'], y=df['fuel_consumption_comb'])
# plt.title('consumption gallon by fuel type')
# plt.show()
# print(df.head())
df.drop(labels=['make', 'model', 'vehicle_class', 'transmission', 'fuel_type', 'fuel_consumption_comb(mpg)'],
            axis=1,
            inplace=True)


y = df['co2_emissions']
df.drop('co2_emissions', axis=1, inplace=True)
# print(df.head())
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=0, shuffle=1)

model = KMeans()
visualizer = KElbowVisualizer(model, k=(4,400))

visualizer.fit(X_train)    # Fit the data to the visualizer
visualizer.show()    # Draw/show/poof the data


# exploratory data analysis visualization and analysis
# plt.figure(figsize=(10, 8))
# sns.histplot(df.mpg)
# plt.show()

# correlation
f, ax = plt.subplots(figsize=[14, 8])
sns.heatmap(df.corr(), fmt=".2f")
ax.set_title("correlation matrix", fontsize=20)
# plt.show()

sns.pairplot(df, diag_kind='kde')
plt.show()

# print(df.corr('spearman'))


# df['acceleration_power_ratio'] = df['acceleration'] / df['horsepower']


# cylinders,displacement,horsepower,weight,acceleration,model year,origin

# engine_size,cylinders,fuel_consumption_city, fuel_consumption_hwy,fuel_consumption_comb(l/100km)
# make,model,vehicle_class,engine_size,cylinders,transmission,fuel_type,fuel_consumption_city,fuel_consumption_hwy,fuel_consumption_comb(l/100km),fuel_consumption_comb(mpg),co2_emissions
# fiat,500l,station wagon - small,1.4,4,m,x,9.3,7.1,8.3,34,194
# 2,4,5.5,6.7,42,181
# bmw,328d xdrive,compact,2,4,a,d,7.6,5.5,6.7,42,181
# fiat,500l,station wagon - small,1.4,4,m,x,9.3,7.1,8.3,34,194
# 1.4,4,9.3,7.1,8.3,34

# input_data = (1.4, 4, 9.3, 7.1, 8.3)  # exp. 194
# input_data = (6.5, 12, 25.2, 14.1, 16.1)  # ferrari 812 superfast  340 pred.
# input_data = (3.0, 6, 10, 8, 9)

# engine_size,cylinders,fuel_consumption_comb(l/100km)
# input_data = (1.4, 4, 8.3)  # exp. 194

input_data = [1.4, 4, 9.3, 7.1, 8.3]  # exp. 194

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)


def rfr_model(df, input_data):
    df.drop(labels=['make', 'model', 'vehicle_class', 'transmission', 'fuel_type', 'fuel_consumption_comb(mpg)'],
            axis=1,
            inplace=True)


    gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': range(3, 8),
            'n_estimators': (10, 50, 100, 1000),
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

    y = df['co2_emissions']
    df.drop('co2_emissions', axis=1, inplace=True)
    # print(df.head())
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=0, shuffle=1)


    grid_result = gsc.fit(X_train, y_train)
    best_params = grid_result.best_params_

    regr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],
                                random_state=False, verbose=False)  # Perform K-Fold CV


    scores = cross_val_score(regr, X_train, y_train, cv=10, scoring='neg_mean_absolute_error')
    print(scores)
    print(best_params["max_depth"])
    print(best_params["n_estimators"])

    #X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=0, shuffle=1)

    #regr = RandomForestRegressor(random_state=0, max_depth=8)
    # regr = SVR(max_iter=5000)

    regr.fit(X_train, y_train)

    # input_data = [1.4, 4, 9.3, 7.1, 8.3]  # exp. 194

    # change the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the numpy array as we are predicting for only on instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = int(regr.predict(input_data_reshaped))

    # uncomment to print the test and train accuracies
    # print("rfr train accuracy", regr.score(X_train, y_train))
    # print("rfr TEST accuracy", regr.score(X_test, y_test))

    return prediction
