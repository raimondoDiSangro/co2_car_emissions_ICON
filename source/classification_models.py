import numpy as np
import pandas as pd
# from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
#from sklearn.svm import SVR
from sklearn.svm import LinearSVR

# support vector regression
from sklearn.tree import DecisionTreeClassifier


def svr_model(df, input_data):
    df.drop(labels=['make', 'model', 'vehicle_class', 'transmission', 'fuel_type', 'fuel_consumption_comb(mpg)'],
            axis=1,
            inplace=True)

    y = df['co2_emissions']
    df.drop('co2_emissions', axis=1, inplace=True)
    # print(df.head())

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=0, shuffle=1)

    # regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    regr = LinearSVR(max_iter=10000)
    regr.fit(X_train, y_train)
    # Pipeline(steps=[('standardscaler', StandardScaler()),
    #                ('svr', SVR(epsilon=0.2))])

    # input_data = [1.4, 4, 9.3, 7.1, 8.3]  # exp. 194

    # change the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the numpy array as we are predicting for only on instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = regr.predict(input_data_reshaped)

    return prediction


def kneigh_model(df, input_data):
    df.drop(labels=['make', 'model', 'vehicle_class', 'transmission', 'fuel_type', 'fuel_consumption_comb(mpg)'],
            axis=1,
            inplace=True)

    y = df['co2_emissions']
    df.drop('co2_emissions', axis=1, inplace=True)
    # print(df.head())

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=0, shuffle=1)

    Kneighbors = KNeighborsClassifier()

    Kneighbors.fit(X_train, y_train)
    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = Kneighbors.predict(input_data_reshaped)

    return prediction


def decision_tree_model(df, input_data):
    df.drop(labels=['make', 'model', 'vehicle_class', 'transmission', 'fuel_type', 'fuel_consumption_comb(mpg)'],
            axis=1,
            inplace=True)

    y = df['co2_emissions']
    df.drop('co2_emissions', axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=0, shuffle=1)

    decision_tree = DecisionTreeClassifier()

    decision_tree.fit(X_train, y_train)
    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = decision_tree.predict(input_data_reshaped)

    return prediction

