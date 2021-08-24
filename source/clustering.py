import numpy as np
from sklearn.cluster import KMeans

import warnings
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# Elbow score = 46
N_CLUSTER = 46


def categorizationkMeans(df, columns_list):
    output = ""
    # scaler = StandardScaler()
    # scaler.fit_transform(data[col_list])
    # data[col_list] = scaler.transform(data[col_list])
    col_list = []
    for elem in columns_list:
        col_list.append(elem)
    col_list.remove('make')
    col_list.remove('model')
    kMeans_model = KMeans(n_clusters=N_CLUSTER, random_state=0)
    df['cluster'] = kMeans_model.fit_predict(df[col_list])
    records = df[df['cluster'] == df['cluster'].iloc[-1]]
    for index, row in records.iterrows():
        if index != -1:
            output += str(row['make']) + ' '
            output += str(row['model']) + ' \n'
            # print(output)
            # 'fuel_consumption_city', 'fuel_consumption_hwy',
            #                   'fuel_consumption_comb'
            output += 'engine size ' + str(row['engine_size']) + ', \n'
            output += 'cylinders ' + str(row['cylinders']) + ', \n'
            output += 'fuel consumption city ' + str(round(row['fuel_consumption_city'], 2)) + ', \n'
            output += 'fuel consumption hwy ' + str(round(row['fuel_consumption_hwy'], 2)) + ', \n'
            output += 'fuel consumption comb ' + str(round(row['fuel_consumption_comb'], 2)) + ', \n'
            output += 'co2 emissions ' + str(round(row['co2_emissions'], 2))
            output += '\n--------------------------\n'
    return output


def clusterkMeans(df, columns_list, values):
    new_column = []
    for col in columns_list:
        if col in values:
            new_column.append(values[col])
        else:
            new_column.append(np.nan)

    # X = df[col_list]
    # X.loc[-1] = new_col
    # output = categorizationkMeans(df, col_list)
    X = df[columns_list]
    X.loc[-1] = new_column
    output = categorizationkMeans(X, columns_list)
    return output
