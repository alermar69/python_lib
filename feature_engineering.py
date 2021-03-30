from warnings import warn

import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import CondensedNearestNeighbour

import category_encoders as ce


def check_missing(data, output_path=None):
    """"
    Проверка пропущенных значений
    """
    result = pd.concat([data.isnull().sum(), data.isnull().mean()], axis=1)
    result = result.rename(index=str, columns={0: 'total missing', 1: 'proportion'})
    if output_path is not None:
        result.to_csv(output_path + 'missing.csv')
        print(output_path, 'missing.csv')
    return result


def drop_missing(data, axis=0):
    """
    Удаление пропущенных значений
    """
    data_copy = data.copy(deep=True)
    data_copy = data_copy.dropna(axis=axis, inplace=False)
    return data_copy


def add_var_denote_NA(data, NA_col=[]):
    """
    Добавление переменной оценки пропущенных значений
    """
    data_copy = data.copy(deep=True)
    for i in NA_col:
        if data_copy[i].isnull().sum() > 0:
            data_copy[i + '_is_NA'] = np.where(data_copy[i].isnull(), 1, 0)
        else:
            warn("Нет пропущенных значений" % i)
    return data_copy


def impute_NA_with_arbitrary(data, impute_value, NA_col=[]):
    """
    Заполнение пропусков выборочным значением
    """
    data_copy = data.copy(deep=True)
    for i in NA_col:
        if data_copy[i].isnull().sum() > 0:
            data_copy[i + '_' + str(impute_value)] = data_copy[i].fillna(impute_value)
        else:
            warn("Нет пропущенных значений" % i)
    return data_copy


def impute_NA_with_avg(data, strategy='mean', NA_col=[]):
    """
    Заполнение пропущенных значений средним/медианой/модой
     """
    data_copy = data.copy(deep=True)
    for i in NA_col:
        if data_copy[i].isnull().sum() > 0:
            if strategy == 'mean':
                data_copy[i + '_impute_mean'] = data_copy[i].fillna(data[i].mean())
            elif strategy == 'median':
                data_copy[i + '_impute_median'] = data_copy[i].fillna(data[i].median())
            elif strategy == 'mode':
                data_copy[i + '_impute_mode'] = data_copy[i].fillna(data[i].mode()[0])
        else:
            warn("Нет пропущенных значений" % i)
    return data_copy


def impute_NA_with_end_of_distribution(data, NA_col=[]):
    """
    Заполнение пропусков значением из "хвоста" распределения
    """
    data_copy = data.copy(deep=True)
    for i in NA_col:
        if data_copy[i].isnull().sum() > 0:
            data_copy[i + '_impute_end_of_distri'] = data_copy[i].fillna(data[i].mean() + 3 * data[i].std())
        else:
            warn("Нет пропущенных значений" % i)
    return data_copy


def impute_NA_with_random(data, NA_col=[], random_state=0):
    """
    Заполнение пропусков случайными значениями
     """
    data_copy = data.copy(deep=True)
    for i in NA_col:
        if data_copy[i].isnull().sum() > 0:
            data_copy[i + '_random'] = data_copy[i]
            random_sample = data_copy[i].dropna().sample(data_copy[i].isnull().sum(), random_state=random_state)
            random_sample.index = data_copy[data_copy[i].isnull()].index
            data_copy.loc[data_copy[i].isnull(), str(i) + '_random'] = random_sample
        else:
            warn("Нет пропущенных значений" % i)
    return data_copy


# %%
def outlier_detect_arbitrary(data, col, upper_fence, lower_fence):
    """
    Детекция с помощью выборочных значений
    """
    para = (upper_fence, lower_fence)
    tmp = pd.concat([data[col] > upper_fence, data[col] < lower_fence], axis=1)
    outlier_index = tmp.any(axis=1)
    print('Количество выбросов в данных:', outlier_index.value_counts()[1])
    print('Доля выбросов:', outlier_index.value_counts()[1] / len(outlier_index))
    return outlier_index, para


def outlier_detect_IQR(data, col, threshold=3):
    """
    Детекция с помощью Интерквартильное расстояние
    """
    IQR = data[col].quantile(0.75) - data[col].quantile(0.25)
    Lower_fence = data[col].quantile(0.25) - (IQR * threshold)
    Upper_fence = data[col].quantile(0.75) + (IQR * threshold)
    para = (Upper_fence, Lower_fence)
    tmp = pd.concat([data[col] > Upper_fence, data[col] < Lower_fence], axis=1)
    outlier_index = tmp.any(axis=1)
    print('Количество выбросов в данных:', outlier_index.value_counts()[1])
    print('Доля выбросов:', outlier_index.value_counts()[1] / len(outlier_index))
    return outlier_index, para


def outlier_detect_mean_std(data, col, threshold=3):
    """
    Детекция с помощью Среднее-среднеквадратичное отклонение
    """
    Upper_fence = data[col].mean() + threshold * data[col].std()
    Lower_fence = data[col].mean() - threshold * data[col].std()
    para = (Upper_fence, Lower_fence)
    tmp = pd.concat([data[col] > Upper_fence, data[col] < Lower_fence], axis=1)
    outlier_index = tmp.any(axis=1)
    print('Количество выбросов в данных:', outlier_index.value_counts()[1])
    print('Доля выбросов:', outlier_index.value_counts()[1] / len(outlier_index))
    return outlier_index, para


def outlier_detect_MAD(data, col, threshold=3.5):
    """
    Детекция с помощью Медианы абсолютного отклонения (MAD)
    """
    median = data[col].median()
    median_absolute_deviation = np.median([np.abs(y - median) for y in data[col]])
    modified_z_scores = pd.Series([0.6745 * (y - median) / median_absolute_deviation for y in data[col]])
    outlier_index = np.abs(modified_z_scores) > threshold
    print('Количество выбросов в данных:', outlier_index.value_counts()[1])
    print('Доля выбросов:', outlier_index.value_counts()[1] / len(outlier_index))
    return outlier_index


def impute_outlier_with_arbitrary(data, outlier_index, value, col=[]):
    """
    Замена выброса выборочным значением
    """
    data_copy = data.copy(deep=True)
    for i in col:
        data_copy.loc[outlier_index, i] = value
    return data_copy


def windsorization(data, col, para, strategy='both'):
    """
    Виндзоризация
    """
    data_copy = data.copy(deep=True)
    if strategy == 'both':
        data_copy.loc[data_copy[col] > para[0], col] = para[0]
        data_copy.loc[data_copy[col] < para[1], col] = para[1]
    elif strategy == 'top':
        data_copy.loc[data_copy[col] > para[0], col] = para[0]
    elif strategy == 'bottom':
        data_copy.loc[data_copy[col] < para[1], col] = para[1]
    return data_copy


def drop_outlier(data, outlier_index):
    """
    Удаление выбросов
    """
    data_copy = data[~outlier_index]
    return data_copy


def impute_outlier_with_avg(data, col, outlier_index, strategy='mean'):
    """
    Замена выбросов средним/медианой/модой
    """
    data_copy = data.copy(deep=True)
    if strategy == 'mean':
        data_copy.loc[outlier_index, col] = data_copy[col].mean()
    elif strategy == 'median':
        data_copy.loc[outlier_index, col] = data_copy[col].median()
    elif strategy == 'mode':
        data_copy.loc[outlier_index, col] = data_copy[col].mode()[0]
    return data_copy


def constant_feature_detect(data, threshold=0.98):
    """
    Константные признаки
    """
    data_copy = data.copy(deep=True)
    quasi_constant_feature = []
    for feature in data_copy.columns:
        predominant = (data_copy[feature].value_counts() / np.float(
            len(data_copy))).sort_values(ascending=False).values[0]
        if predominant >= threshold:
            quasi_constant_feature.append(feature)
    print(len(quasi_constant_feature), 'константные переменные')
    return quasi_constant_feature


def corr_feature_detect(data, threshold=0.8):
    """
    Корреляционная фильтрация

    corr = corr_feature_detect(data=X_train,threshold=0.9)

    for i in corr:
        print(i,"n")
    """
    corrmat = data.corr()
    corrmat = corrmat.abs().unstack()
    corrmat = corrmat.sort_values(ascending=False)
    corrmat = corrmat[corrmat >= threshold]
    corrmat = corrmat[corrmat < 1]
    corrmat = pd.DataFrame(corrmat).reset_index()
    corrmat.columns = ['feature1', 'feature2', 'corr']

    grouped_feature_ls = []
    correlated_groups = []

    for feature in corrmat.feature1.unique():
        if feature not in grouped_feature_ls:
            correlated_block = corrmat[corrmat.feature1 == feature]
            grouped_feature_ls = grouped_feature_ls + list(
                correlated_block.feature2.unique()) + [feature]

            correlated_groups.append(correlated_block)
    return correlated_groups


def mutual_info(X, y, select_k=10):
    """
    Взаимная информация
    """
    if select_k >= 1:
        sel_ = SelectKBest(mutual_info_classif, k=select_k).fit(X, y)
        col = X.columns[sel_.get_support()]

    elif 0 < select_k < 1:
        sel_ = SelectPercentile(mutual_info_classif, percentile=select_k * 100).fit(X, y)
        col = X.columns[sel_.get_support()]

    else:
        raise ValueError("select_k должно быть положительным значением")

    return col


def chi_square_test(X, y, select_k=10):
    """
    Хи-квадрат тест
    """
    if select_k >= 1:
        sel_ = SelectKBest(chi2, k=select_k).fit(X, y)
        col = X.columns[sel_.get_support()]
    elif 0 < select_k < 1:
        sel_ = SelectPercentile(chi2, percentile=select_k * 100).fit(X, y)
        col = X.columns[sel_.get_support()]
    else:
        raise ValueError("select_k должно быть положительным значением")

    return col


def univariate_roc_auc(X_train, y_train, X_test, y_test, threshold=0.8):
    """
    Одномерный ROC-AUC анализ
    """
    roc_values = []
    for feature in X_train.columns:
        clf = DecisionTreeClassifier()
        clf.fit(X_train[feature].to_frame(), y_train)
        y_scored = clf.predict_proba(X_test[feature].to_frame())
        roc_values.append(roc_auc_score(y_test, y_scored[:, 1]))
    roc_values = pd.Series(roc_values)
    roc_values.index = X_train.columns
    print(roc_values.sort_values(ascending=False))
    print(len(roc_values[roc_values > threshold]), len(X_train.columns))
    keep_col = roc_values[roc_values > threshold]
    return keep_col


def univariate_mse(X_train, y_train, X_test, y_test, threshold=0.8):
    """
    Одномерный MSE анализ
    """
    mse_values = []
    for feature in X_train.columns:
        clf = DecisionTreeRegressor()
        clf.fit(X_train[feature].to_frame(), y_train)
        y_scored = clf.predict(X_test[feature].to_frame())
        mse_values.append(mean_squared_error(y_test, y_scored))
    mse_values = pd.Series(mse_values)
    mse_values.index = X_train.columns
    print(mse_values.sort_values(ascending=False))
    print(len(mse_values[mse_values > threshold]), len(X_train.columns))
    keep_col = mse_values[mse_values > threshold]
    return keep_col
