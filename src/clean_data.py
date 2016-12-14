"""
prepare and clean data
"""
from __future__ import division
import pandas as pd
from tqdm import tqdm
import numpy as np
from sframe import SFrame
from sklearn.preprocessing import LabelEncoder


def create_joined():
    train = pd.read_csv('../data/train_ver2.csv', na_values=[' NA', '     NA', 'NA', '         NA'], dtype={'renta': float})
    test = pd.read_csv('../data/test_ver2.csv', na_values=[' NA', '     NA', 'NA', '         NA'], dtype={'renta': float})

    train.loc[train['age'] > 118] = np.nan
    test.loc[test['age'] > 118] = np.nan

    train = train[~train['fecha_dato'].isin(['2015-07-28',
                                             '2015-08-28',
                                             '2015-09-28',
                                             '2015-10-28',
                                             '2015-11-28',
                                             '2015-12-28'])]

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

    train_cut['fecha_dato'] = (train_cut['fecha_dato']
                               .str
                               .replace('2015-', '')
                               .str
                               .replace('-28', ''))

    test_cut['fecha_dato'] = (test_cut['fecha_dato']
                              .str
                              .replace('2016-', '')
                              .str
                              .replace('-28', ''))

    train_target = train_cut[train_cut['fecha_dato'] == '06']

    train_X = train_cut[train_cut['fecha_dato'] != '06']

    to_drop = ['age',
               'sexo',
               'canal_entrada',
               'conyuemp',
               'segmento',
               'nomprov',
               'pais_residencia',
               'indresi',
               'indext']

    train_X = train_X.drop(to_drop, 1)
    test_cut = test_cut.drop(to_drop, 1)

    temp_train = train_X[train_X['fecha_dato'] == '01'].drop('fecha_dato', 1)
    temp_test = test_cut[test_cut['fecha_dato'] == '01'].drop('fecha_dato', 1)

    old_columns = [x for x in temp_train.columns if x != 'ncodpers']
    new_columns = [x + '_01' for x in old_columns]

    temp_train = temp_train.rename(columns=dict(zip(old_columns, new_columns)))
    temp_test = temp_test.rename(columns=dict(zip(old_columns, new_columns)))

    for month in tqdm(['02', '03', '04', '05']):
        to_merge_train = train_X[train_X['fecha_dato'] == month].drop('fecha_dato', 1)
        to_merge_test = test_cut[test_cut['fecha_dato'] == month].drop('fecha_dato', 1)

        old_columns = [x for x in to_merge_train.columns if x != 'ncodpers']
        new_columns = [x + '_' + month for x in old_columns]
        to_merge_train = to_merge_train.rename(columns=dict(zip(old_columns, new_columns)))
        to_merge_test = to_merge_test.rename(columns=dict(zip(old_columns, new_columns)))

        temp_train = temp_train.merge(to_merge_train, on='ncodpers', how='outer')
        temp_test = temp_test.merge(to_merge_test, on='ncodpers', how='outer')

    X = train_target.drop('fecha_dato', 1).merge(temp_train, on='ncodpers', how='left')

    X_test = test.drop('fecha_dato', 1).merge(temp_test, on='ncodpers', how='left')

    return pd.concat([X, X_test])


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
    temp = create_joined()

    for column in target_variables():
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
    train = joined[joined[target_variables()].notnull().any(axis=1)]
    test = joined[joined[target_variables()].isnull().any(axis=1)].drop(target_variables(), 1)

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

    for column in tqdm(train.columns[train.dtypes == 'object']):
        le = LabelEncoder()
        train[column] = le.fit_transform(train[column])
        test[column] = le.fit_transform(test[column])

    train = stack_train(train)

    return train, test


if __name__ == '__main__':
    import os
    os.environ['HDF5_DISABLE_VERSION_CHECK'] = str(2)
    label_encoded(True)
