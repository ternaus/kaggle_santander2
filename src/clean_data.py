"""
prepare and clean data
"""
from __future__ import division
import pandas as pd
from tqdm import tqdm
import numpy as np
from sframe import SFrame
from sklearn.preprocessing import LabelEncoder


def create_joined(cached=False):
    if cached:
        try:
            train = pd.read_hdf('../data/train_raw.h5')
            test = pd.read_hdf('../data/test_raw.h5')
            return train, test
        except:
            pass

    missing_values = [' NA', '     NA', 'NA', '         NA', -99]
    dtype = {'ncodpers': int,
             'fecha_dato': str,
             'sexo': str,
             'ult_fec_cli_1t': str,
             'indext': str,
             'canal_entrada': str,
             'pais_residencia': str}

    train = pd.read_csv('../data/train_ver2.csv', na_values=missing_values, dtype=dtype)
    test = pd.read_csv('../data/test_ver2.csv', na_values=missing_values, dtype=dtype)

    train = train[~train['fecha_dato'].isin(['2015-07-28',
                                             '2015-08-28',
                                             '2015-09-28',
                                             '2015-10-28',
                                             '2015-11-28',
                                             '2015-12-28'])]

    train['age'] = train['age'].apply(get_age)
    test['age'] = test['age'].apply(get_age)

    train['antiguedad'] = train['antiguedad'].apply(get_seniority)
    test['antiguedad'] = test['antiguedad'].apply(get_seniority)

    train['renta'] = train['renta'].apply(get_rent)
    test['renta'] = test['renta'].apply(get_rent)

    # Rent can have trend and it is important feature => normalize it by median
    g_rent_train = train.groupby('fecha_dato')['renta'].median()
    for date in tqdm(train['fecha_dato'].unique()):
        train.loc[train['fecha_dato'] == date, 'renta'] /= g_rent_train[date]

    test['renta'] /= train['renta'].median()

    train_cut = train[train['fecha_dato'].isin(['2015-01-28',
                                                '2015-02-28',
                                                '2015-03-28',
                                                '2015-04-28',
                                                '2015-05-28',
                                                '2015-06-28'])]

    test_cut = train[train['fecha_dato'].isin(['2016-01-28',
                                               '2016-02-28',
                                               '2016-03-28',
                                               '2016-04-28',
                                               '2016-05-28'])]

    train_cut['fecha_dato'] = pd.to_datetime(train_cut['fecha_dato'])
    train_cut['fecha_alta'] = pd.to_datetime(train_cut['fecha_alta'])

    train_cut['month'] = train_cut['fecha_dato'].dt.month

    test_cut['fecha_dato'] = pd.to_datetime(test_cut['fecha_dato'])
    test_cut['fecha_alta'] = pd.to_datetime(test_cut['fecha_alta'])

    test_cut['month'] = test_cut['fecha_dato'].dt.month

    train_target = train_cut[train_cut['month'] == 6]

    train_target.loc[:, 'days'] = (train_target['fecha_dato'] - train_target['fecha_alta']).dt.days
    test_cut.loc[:, 'days'] = (test_cut['fecha_dato'] - test_cut['fecha_alta']).dt.days

    train_X = train_cut[train_cut['month'] != 6]

    to_drop = ['age',
               'sexo',
               'canal_entrada',
               'conyuemp',
               'segmento',
               'nomprov',
               'cod_prov',
               'pais_residencia',
               'indresi',
               'indext',
               'indfall',
               'fecha_alta',
               'fecha_dato',
               ]

    train_X = train_X.drop(to_drop, 1)
    test_cut = test_cut.drop(to_drop, 1)

    temp_train = train_X[train_X['month'] == 1].drop('month', 1)
    temp_test = test_cut[test_cut['month'] == 1].drop('month', 1)

    old_columns = [x for x in temp_train.columns if x != 'ncodpers']
    new_columns = [x + '_01' for x in old_columns]

    temp_train = temp_train.rename(columns=dict(zip(old_columns, new_columns)))
    temp_test = temp_test.rename(columns=dict(zip(old_columns, new_columns)))

    for month in tqdm([2, 3, 4, 5]):
        to_merge_train = train_X[train_X['month'] == month].drop('month', 1)
        to_merge_test = test_cut[test_cut['month'] == month].drop('month', 1)

        old_columns = [x for x in to_merge_train.columns if x != 'ncodpers']

        new_columns = [str(x) + '_0' + str(month)for x in old_columns]

        to_merge_train = to_merge_train.rename(columns=dict(zip(old_columns, new_columns)))
        to_merge_test = to_merge_test.rename(columns=dict(zip(old_columns, new_columns)))

        temp_train = temp_train.merge(to_merge_train, on='ncodpers', how='outer')
        temp_test = temp_test.merge(to_merge_test, on='ncodpers', how='outer')

    X = train_target.merge(temp_train, on='ncodpers', how='left')

    X_test = test.merge(temp_test, on='ncodpers', how='left')

    to_drop = ['fecha_dato', 'fecha_alta']

    X_test = X_test.drop(to_drop, 1)
    X = X.drop(to_drop, 1)

    assert X_test.shape[0] == 929615

    print list(X.columns)
    print list(X_test.columns)

    X.to_hdf('../data/train_raw.h5', 'Table')
    X_test.to_hdf('../data/test_raw.h5', 'Table')

    print X.shape, X_test.shape

    return X, X_test


def target_variables():
    return ['ind_ahor_fin_ult1',
             'ind_aval_fin_ult1',
             'ind_cco_fin_ult1',
             'ind_cder_fin_ult1',
             'ind_cno_fin_ult1',
             'ind_ctju_fin_ult1',
             'ind_ctma_fin_ult1',
             'ind_ctop_fin_ult1',
             'ind_ctpp_fin_ult1',
             'ind_deco_fin_ult1',
             'ind_dela_fin_ult1',
             'ind_deme_fin_ult1',
             'ind_ecue_fin_ult1',
             'ind_fond_fin_ult1',
             'ind_hip_fin_ult1',
             'ind_nom_pens_ult1',
             'ind_nomina_ult1',
             'ind_plan_fin_ult1',
             'ind_pres_fin_ult1',
             'ind_reca_fin_ult1',
             'ind_recibo_ult1',
             'ind_tjcr_fin_ult1',
             'ind_valo_fin_ult1',
             'ind_viv_fin_ult1']


def filled_zero_joined():
    X, X_test = create_joined(cached=True)

    temp = pd.concat([X, X_test])

    for column in tqdm(target_variables()):
        temp[column + '_01'] = temp[column + '_01'].fillna(0)
        for month in ['02', '03', '04', '05']:
            current_column = column + '_' + month
            old_column = column + '_0' + str(int(month) - 1)
            null_index = temp[current_column].isnull()
            temp.loc[null_index, current_column] = temp.loc[null_index, old_column]
            assert temp[current_column].isnull().sum() == 0
    return temp


def get_train_test(cached=False):
    if cached:
        try:
            train = pd.read_hdf('../data/train.h5')
            test = pd.read_hdf('../data/test.h5')
            return train, test
        except:
            pass

    joined = filled_zero_joined()

    test_index = joined[target_variables()].isnull().all(axis=1)

    print test_index.sum()

    train = joined[~test_index]
    test = joined[test_index].drop(target_variables(), 1)

    assert test.shape[0] == 929615

    df = pd.DataFrame()
    for column in target_variables():
        df[column] = (train[column] - train[column + '_05']) > 0

    train = train[(df.abs().sum(axis=1) != 0)]
    train.to_hdf('../data/train.h5', 'Table')
    test.to_hdf('../data/test.h5', 'Table')

    return train, test


def get_target_matrix(train):
    target_previous = train[[x + '_05' for x in target_variables()]]
    target_current = train[target_variables()]
    return (target_current.values > target_previous.values).astype(int)


def stack_train(train):
    target = get_target_matrix(train)
    target_names = np.array(target_variables())

    def helper(x):
        return target_names[target[x] == 1]

    result = map(helper, range(target.shape[0]))
    df = pd.DataFrame({'ncodpers': train['ncodpers'], 'added_products': result})
    ncod_prod_mapping = SFrame(df).stack('added_products', new_column_name='added_product').to_dataframe()

    return train.merge(ncod_prod_mapping, on='ncodpers')


def label_encoded(cached):
    train, test = get_train_test(cached)

    print test.shape

    for column in tqdm(train.columns[train.dtypes == 'object']):
        le = LabelEncoder()
        set_train = set(train[column].unique())
        set_test = set(test[column].unique())
        remove_train = set_train - set_test
        remove_test = set_test - set_train

        remove = remove_train.union(remove_test)

        def helper(x):
            if x in remove:
                return np.nan
            return x

        train[column] = train[column].apply(helper)
        test[column] = test[column].apply(helper)

        train[column] = train[column].fillna('')
        test[column] = test[column].fillna('')

        train[column] = le.fit_transform(train[column])
        test[column] = le.transform(test[column])

    train = stack_train(train)

    return train, test


def get_age(age):
    mean_age = 40.0
    min_age = 20.0
    max_age = 90.0
    range_age = max_age - min_age

    if np.isnan(age):
        age = mean_age
    else:
        age = float(age)
        if age < min_age:
            age = min_age
        elif age > max_age:
            age = max_age

    return round((age - min_age) / range_age, 4)


def get_seniority(cust_seniority):
    min_value = 0.
    max_value = 256.0
    range_value = max_value - min_value
    missing_value = -99999.
    if np.isnan(cust_seniority):
        cust_seniority = missing_value
    else:
        cust_seniority = min(max_value, max(min_value, cust_seniority))
    return round((cust_seniority - min_value) / range_value, 4)


def get_rent(rent):
    min_value = 0.0
    max_value = 1500000.0
    # missing_value = -99999
    if not np.isnan(rent):
        rent = min(max_value, max(min_value, rent))
    return rent


def submission(preds, X_test):
    classes = np.array([x for x in preds.columns if x != 'ncodpers'])
    prediction = preds[classes].values
    previous = X_test[[x + '_05' for x in classes]].values
    temp = prediction - previous

    def helper(x):
        return ' '.join(classes[np.argsort(temp[x])[:-8:-1]])

    result = map(lambda x: helper(x), range(temp.shape[0]))

    return pd.DataFrame({'ncodpers': preds['ncodpers'].values, 'added_products': result})


if __name__ == '__main__':
    import os
    os.environ['HDF5_DISABLE_VERSION_CHECK'] = str(2)
    create_joined(cached=False)
