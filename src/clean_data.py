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

    missing_values = [' NA', '     NA', 'NA', '         NA', -99, -999999, 101850]
    dtype = {'ncodpers': np.int32,
             'renta': np.float32,
             'age': np.float32,
             'fecha_dato': str,
             'sexo': str,
             'ult_fec_cli_1t': str,
             'indext': str,
             'canal_entrada': str,
             'pais_residencia': str,
             'ind_ahor_fin_ult1': np.float32,
            'ind_aval_fin_ult1': np.float32,
            'ind_cco_fin_ult1': np.float32,
            'ind_cder_fin_ult1': np.float32,
            'ind_cno_fin_ult1': np.float32,
            'ind_ctju_fin_ult1': np.float32,
            'ind_ctma_fin_ult1': np.float32,
            'ind_ctop_fin_ult1': np.float32,
            'ind_ctpp_fin_ult1': np.float32,
            'ind_deco_fin_ult1': np.float32,
            'ind_dela_fin_ult1': np.float32,
            'ind_deme_fin_ult1': np.float32,
            'ind_ecue_fin_ult1': np.float32,
            'ind_fond_fin_ult1': np.float32,
            'ind_hip_fin_ult1': np.float32,
            'ind_nom_pens_ult1': np.float32,
            'ind_nomina_ult1': np.float32,
            'ind_plan_fin_ult1': np.float32,
            'ind_pres_fin_ult1': np.float32,
            'ind_reca_fin_ult1': np.float32,
            'ind_recibo_ult1': np.float32,
            'ind_tjcr_fin_ult1': np.float32,
            'ind_valo_fin_ult1': np.float32,
            'ind_viv_fin_ult1': np.float32
             }

    train = pd.read_csv('../data/train_ver2.csv', na_values=missing_values, dtype=dtype)
    test = pd.read_csv('../data/test_ver2.csv', na_values=missing_values, dtype=dtype)

    train = train[~train['fecha_dato'].isin(['2015-07-28',
                                             '2015-08-28',
                                             '2015-09-28',
                                             '2015-10-28',
                                             '2015-11-28',
                                             '2015-12-28'])]

    train['tipodom'] = train['tipodom'].fillna(0).astype(int)
    test['tipodom'] = test['tipodom'].fillna(0).astype(int)

    print train['indrel_1mes'].value_counts()
    train['indrel_1mes'] = train['indrel_1mes'].map({1: 1, 2: 2, 3: 3, 4: 4, 5: 5, '1.0': 1, '1': 1, '2.0': 2, '2': 2, '3.0': 3, '3': 3, '4.0': 4, '4': 4, 'P': 5})
    print train['indrel_1mes'].isnull().sum()
    print train['indrel_1mes'].value_counts()
    print train['indrel_1mes'].isnull().sum()
    test['indrel_1mes'] = test['indrel_1mes'].map({1: 1, 2: 2, 3: 3, 4: 4, 5: 5, '1.0': 1, '1': 1, '2.0': 2, '2': 2, '3.0': 3, '3': 3, '4.0': 4, '4': 4, 'P': 5})

    train['age'] = train['age'].apply(get_age)
    test['age'] = test['age'].apply(get_age)

    train['nan_age'] = train['age'].isnull().astype(int)
    test['nan_age'] = test['age'].isnull().astype(int)

    mean_age = 40
    train['age'] = train['age'].fillna(mean_age)
    train['age'] = train['age'].fillna(mean_age)

    train['antiguedad'] = train['antiguedad'].apply(get_seniority)
    test['antiguedad'] = test['antiguedad'].apply(get_seniority)

    train['renta'] = train['renta'].apply(get_rent)
    test['renta'] = test['renta'].apply(get_rent)

    sexo_mode = train['sexo'].mode()[0]

    train['sexo'] = train['sexo'].fillna(sexo_mode)
    test['sexo'] = test['sexo'].fillna(sexo_mode)

    train['canal_entrada'] = train['canal_entrada'].fillna('KAT')
    test['canal_entrada'] = test['canal_entrada'].fillna('KAT')

    train.loc[train["ind_nuevo"].isnull(), "ind_nuevo"] = 1
    test.loc[test["ind_nuevo"].isnull(), "ind_nuevo"] = 1

    train.loc[train.indrel.isnull(), "indrel"] = -1
    test.loc[test.indrel.isnull(), "indrel"] = -1

    test['conyuemp'] = test['conyuemp'].fillna('')
    train['conyuemp'] = train['conyuemp'].fillna('')

    train['segmento'] = train['segmento'].fillna('02 - PARTICULARES')
    test['segmento'] = test['segmento'].fillna('02 - PARTICULARES')

    train['nomprov'] = train['nomprov'].fillna('UNKNOWN')
    test['nomprov'] = test['nomprov'].fillna('UNKNOWN')

    # Filling missing values in train
    g_rent_train = train.groupby('nomprov')['renta'].median()

    g_rent_test = test.groupby(['nomprov'])['renta'].median()

    train.loc[train['renta'].isnull(), 'renta'] = train.loc[train['renta'].isnull(), 'nomprov'].map(g_rent_train)
    test.loc[test['renta'].isnull(), 'renta'] = test.loc[test['renta'].isnull(), 'nomprov'].map(g_rent_test)

    fill_rent = train['renta'].median()
    train['renta'] = train['renta'].fillna(fill_rent)
    test['renta'] = test['renta'].fillna(fill_rent)

    # Rent can have trend and it is important feature => normalize it by median
    g_rent_train = train.groupby('fecha_dato')['renta'].median()
    for date in tqdm(train['fecha_dato'].unique()):
        train.loc[train['fecha_dato'] == date, 'renta'] /= g_rent_train[date]

    test['renta'] /= test['renta'].median()

    train['marriage_index'] = train.apply(lambda x: get_marriage_index(x), 1)
    test['marriage_index'] = test.apply(lambda x: get_marriage_index(x), 1)

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

    train_cut.loc[:, 'fecha_dato'] = pd.to_datetime(train_cut['fecha_dato'])
    train_cut.loc[:, 'fecha_alta'] = pd.to_datetime(train_cut['fecha_alta'])
    train_cut.loc[:, 'ult_fec_cli_1t'] = pd.to_datetime(train_cut['ult_fec_cli_1t'])

    train_cut.loc[:, 'month'] = train_cut['fecha_dato'].dt.month

    test.loc[:, 'fecha_dato'] = pd.to_datetime(test['fecha_dato'])
    test.loc[:, 'fecha_alta'] = pd.to_datetime(test['fecha_alta'])
    test.loc[:, 'ult_fec_cli_1t'] = pd.to_datetime(test['ult_fec_cli_1t'])

    # test_cut.loc[:, 'fecha_alta'] = pd.to_datetime(test_cut['fecha_alta'])
    test_cut.loc[:, 'fecha_dato'] = pd.to_datetime(test_cut['fecha_dato'])

    test_cut.loc[:, 'month'] = test_cut['fecha_dato'].dt.month

    train_target = train_cut[train_cut['month'] == 6]

    train_target.loc[:, 'days'] = (train_target['fecha_dato'] - train_target['fecha_alta']).dt.days
    train_target.loc[:, 'days_primary'] = (train_target['fecha_dato'] - train_target['ult_fec_cli_1t']).dt.days
    train_target.loc[:, 'alta_month'] = train_target['fecha_alta'].dt.month
    train_target.loc[:, 'ult_month'] = train_target['ult_fec_cli_1t'].dt.month
    # train_target.loc[:, 'alta_year'] = train_target['fecha_alta'].dt.year
    # train_target.loc[:, 'ult_year'] = train_target['ult_fec_cli_1t'].dt.year

    test.loc[:, 'days'] = (test['fecha_dato'] - test['fecha_alta']).dt.days
    test.loc[:, 'days_primary'] = (test['fecha_dato'] - test['ult_fec_cli_1t']).dt.days
    test.loc[:, 'alta_month'] = test['fecha_alta'].dt.month
    test.loc[:, 'ult_month'] = test['ult_fec_cli_1t'].dt.month
    # test.loc[:, 'alta_year'] = test['fecha_alta'].dt.year
    # test.loc[:, 'ult_year'] = test['ult_fec_cli_1t'].dt.year

    # Add statistical features to the last column
    # to_encode = ['sexo', 'segmento', 'pais_residencia', 'nomprov', 'canal_entrada']
    #
    # for column in to_encode:
    #     train_target[column + '_mean'] = train_target[column].map(train_target.groupby(column)['renta'].mean())
    #     train_target[column + '_median'] = train_target[column].map(train_target.groupby(column)['renta'].median())
    #     train_target[column + '_std'] = train_target[column].map(train_target.groupby(column)['renta'].std())
    #     train_target[column + '_min'] = train_target[column].map(train_target.groupby(column)['renta'].min())
    #     train_target[column + '_max'] = train_target[column].map(train_target.groupby(column)['renta'].max())
    #
    #     test[column + '_mean'] = test[column].map(test.groupby(column)['renta'].mean())
    #     test[column + '_median'] = test[column].map(test.groupby(column)['renta'].median())
    #     test[column + '_std'] = test[column].map(test.groupby(column)['renta'].std())
    #     test[column + '_min'] = test[column].map(test.groupby(column)['renta'].min())
    #     test[column + '_max'] = test[column].map(test.groupby(column)['renta'].max())

    train_X = train_cut[train_cut['month'] != 6]

    to_drop = [
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
               'ult_fec_cli_1t',
               'ind_nuevo',
               'tipodom',
                'age',
                'nan_age',
               'antiguedad',
               'marriage_index'
               ] #+ to_encode

    train_X = train_X.drop(to_drop, 1)
    test_cut = test_cut.drop(to_drop, 1)

    temp_train = train_X[train_X['month'] == 1].drop('month', 1)
    temp_train['num_products'] = temp_train[target_variables()].sum(axis=1)

    temp_test = test_cut[test_cut['month'] == 1].drop('month', 1)
    temp_test['num_products'] = temp_test[target_variables()].sum(axis=1)

    old_columns = [x for x in temp_train.columns if x != 'ncodpers']
    new_columns = [x + '_01' for x in old_columns]

    temp_train = temp_train.rename(columns=dict(zip(old_columns, new_columns)))
    temp_test = temp_test.rename(columns=dict(zip(old_columns, new_columns)))

    for month in tqdm([2, 3, 4, 5]):
        to_merge_train = train_X[train_X['month'] == month].drop('month', 1)
        to_merge_test = test_cut[test_cut['month'] == month].drop('month', 1)

        to_merge_train['num_products'] = to_merge_train[target_variables()].sum(axis=1)
        to_merge_test['num_products'] = to_merge_test[target_variables()].sum(axis=1)

        old_columns = [x for x in to_merge_train.columns if x != 'ncodpers']

        new_columns = [str(x) + '_0' + str(month)for x in old_columns]

        to_merge_train = to_merge_train.rename(columns=dict(zip(old_columns, new_columns)))
        to_merge_test = to_merge_test.rename(columns=dict(zip(old_columns, new_columns)))

        temp_train = temp_train.merge(to_merge_train, on='ncodpers', how='outer')
        temp_test = temp_test.merge(to_merge_test, on='ncodpers', how='outer')

    X = train_target.merge(temp_train, on='ncodpers', how='left')

    # Let's drop train rows that do not have data

    to_drop_columns = [x + '_05' for x in target_variables()] + [x + '_04' for x in target_variables()]

    to_drop_index = (X[to_drop_columns].isnull().all(axis=1))
    print 'we drop num rows = ', to_drop_index.sum()
    X = X[~to_drop_index]

    X_test = test.merge(temp_test, on='ncodpers', how='left')

    to_drop = ['fecha_dato', 'fecha_alta', 'ult_fec_cli_1t'] #+ to_encode

    X_test = X_test.drop(to_drop, 1)
    X = X.drop(to_drop, 1)

    # Add renta comparisons
    X['renta65'] = X['renta'] - X['renta_05']
    X['renta64'] = X['renta'] - X['renta_04']
    X['renta63'] = X['renta'] - X['renta_03']
    X['renta62'] = X['renta'] - X['renta_02']
    X['renta61'] = X['renta'] - X['renta_01']

    X_test['renta65'] = X_test['renta'] - X_test['renta_05']
    X_test['renta64'] = X_test['renta'] - X_test['renta_04']
    X_test['renta63'] = X_test['renta'] - X_test['renta_03']
    X_test['renta62'] = X_test['renta'] - X_test['renta_02']
    X_test['renta61'] = X_test['renta'] - X_test['renta_01']

    X['indrel65'] = X['indrel'] - X['indrel_05']
    X['indrel64'] = X['indrel'] - X['indrel_04']
    X['indrel63'] = X['indrel'] - X['indrel_03']
    X['indrel62'] = X['indrel'] - X['indrel_02']
    X['indrel61'] = X['indrel'] - X['indrel_01']

    X_test['indrel65'] = X_test['indrel'] - X_test['indrel_05']
    X_test['indrel64'] = X_test['indrel'] - X_test['indrel_04']
    X_test['indrel63'] = X_test['indrel'] - X_test['indrel_03']
    X_test['indrel62'] = X_test['indrel'] - X_test['indrel_02']
    X_test['indrel61'] = X_test['indrel'] - X_test['indrel_01']


    assert X_test.shape[0] == 929615

    print sorted(list(X.columns))
    print sorted(list(X_test.columns))

    X.to_hdf('../data/train_raw.h5', 'Table')
    X_test.to_hdf('../data/test_raw.h5', 'Table')

    print X.shape, X_test.shape
    print X.isnull().sum()
    print X_test.isnull().sum()
    print X.var()

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

    for column in ['ind_nom_pens_ult1',
                   'ind_nomina_ult1',
                   'ind_recibo_ult1',
                   'ind_cco_fin_ult1',
                   'ind_ctma_fin_ult1',
                   'ind_ctju_fin_ult1',
                   'ind_tjcr_fin_ult1',
                   'ind_reca_fin_ult1']:
        for month in [5, 4, 3]:
            joined[column + "_back_" + str(month)] = joined[column + '_0' + str(month)] - joined[column + '_0' + str(month - 1)]

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

        # print column
        # print train[column].value_counts()
        # print test[column].value_counts()

        fill_mode = train[column].mode()[0]
        train[column] = train[column].fillna(fill_mode)
        test[column] = test[column].fillna(fill_mode)

        mapping = get_mapping(train, column)

        train[column] = train[column].map(mapping)
        test[column] = test[column].map(mapping)

    train = stack_train(train)

    return train, test


def get_age(age):
    # mean_age = 40.0
    # min_age = 20.0
    max_age = 90.0

    if not np.isnan(age):
        return min(age, max_age)

    return np.nan


def get_seniority(cust_seniority):
    min_value = 0.
    max_value = 256.0

    if np.isnan(cust_seniority):
        cust_seniority = min_value
    else:
        cust_seniority = min(max_value, max(min_value, cust_seniority))
    return cust_seniority


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


# def get_mapping(train, column):
#     df = train.groupby(column).agg(
#         {'renta': {'median': np.mean, 'count': np.count_nonzero, 'var': np.var}}).reset_index()
#     df = pd.DataFrame(
#         {column: df[column], 'count': df['renta']['count'], 'median': df['renta']['median'], 'var': df['renta']['var']})
#     df['var'] /= train['renta'].var()
#     full_median = train['renta'].median()
#
#     def helper(x):
#         n = x['count']
#         m = x['var']
#         ld = n / (n + m)
#         return ld * x['median'] / full_median + (1 - ld)
#
#     df = df[df['median'].notnull()]
#
#     df['value'] = df.apply(helper, 1)
#     return dict(zip(df[column].values, df['value'].values))


def get_mapping(train, column):
    # print column
    # print train['num_products_05'].mean(), train['num_products_05'].count_nonzero(), train['num_products_05'].var()
    df = train.groupby(column).agg({'num_products_05': {'mean': np.mean, 'count': np.count_nonzero, 'var': np.var}}).reset_index()
    # print df
    df = pd.DataFrame(
        {column: df[column], 'count': df['num_products_05']['count'],
         'mean': df['num_products_05']['mean'],
         'var': df['num_products_05']['var']})

    df['var'] /= train['num_products_05'].var()
    full_mean = train['num_products_05'].mean()

    def helper(x):
        n = x['count']
        m = x['var']
        ld = n / (n + m)

        return ld * x['mean'] + (1 - ld) * full_mean

    df = df[df['mean'].notnull()]

    df['value'] = df.apply(helper, 1)
    print dict(zip(df[column].values, df['value'].values))
    return dict(zip(df[column].values, df['value'].values))


def get_marriage_index(row):
    age = row['age']
    sex = row['sexo']
    income = row['renta']
    marriage_age = 28
    modifier = 0
    if sex == 'V':
        modifier += -2
    if income <= 101850:
        modifier += -1

    marriage_age_mod = marriage_age + modifier

    return int(age > marriage_age_mod)


if __name__ == '__main__':
    import os
    os.environ['HDF5_DISABLE_VERSION_CHECK'] = str(2)
    create_joined(cached=False)
