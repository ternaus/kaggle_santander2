"""
prepare and clean data
"""
from __future__ import division
import pandas as pd


def create_joined():
    train = pd.read_csv('../data/train_ver2.csv')
    test = pd.read_csv('../data/test_ver2.csv')

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

    temp_train = train_X[train_X['fecha_dato'] == '01'].drop('fecha_dato', 1)
    temp_test = test_cut[test_cut['fecha_dato'] == '01'].drop('fecha_dato', 1)

    old_columns = [x for x in temp_train.columns if x not in ['ncodpers', 'fecha_dato']]
    new_columns = [x + '_01' for x in old_columns]

    temp_train = temp_train.rename(columns=dict(zip(old_columns, new_columns)))
    temp_test = temp_test.rename(columns=dict(zip(old_columns, new_columns)))

    for month in tqdm(['02', '03', '04', '05']):
        to_merge_train = train_X[train_X['fecha_dato'] == month].drop('fecha_dato', 1)
        to_merge_test = test_cut[test_cut['fecha_dato'] == month].drop('fecha_dato', 1)

        old_columns = [x for x in to_merge_train.columns if x not in ['ncodpers', 'fecha_dato']]
        new_columns = [x + '_' + month for x in old_columns]
        to_merge_train = to_merge_train.rename(columns=dict(zip(old_columns, new_columns)))
        to_merge_test = to_merge_test.rename(columns=dict(zip(old_columns, new_columns)))

        temp_train = temp_train.merge(to_merge_train, on='ncodpers', how='outer')
        temp_test = temp_test.merge(to_merge_test, on='ncodpers', how='outer')

    X = train_target.drop('fecha_dato', 1).merge(temp_train, on='ncodpers', how='left')

    X_test = test.drop('fecha_dato', 1).merge(temp_test, on='ncodpers', how='left')

    return pd.concat([X, X_test])