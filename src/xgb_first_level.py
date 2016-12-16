"""
create oof prediction using xgb
"""

from __future__ import division
import clean_data
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import log_loss
import pandas as pd
import datetime


features = [
#     'added_product',
 'age',
 'age43',
 'age53',
 'age54',
 'age61',
 'age62',
 'age63',
 'age64',
 'age65',
 'age_01',
 'age_02',
 'age_03',
 'age_04',
 'age_05',
 'antiguedad',
 'canal_entrada_max',
 'canal_entrada_mean',
 'canal_entrada_median',
 'canal_entrada_min',
 'canal_entrada_std',
 'cod_prov',
 'conyuemp',
 'days',
 'days_primary',
 'ind_actividad_cliente',
 'ind_actividad_cliente_01',
 'ind_actividad_cliente_02',
 'ind_actividad_cliente_03',
 'ind_actividad_cliente_04',
 'ind_actividad_cliente_05',
#  'ind_ahor_fin_ult1',
 'ind_ahor_fin_ult1_01',
 'ind_ahor_fin_ult1_02',
 'ind_ahor_fin_ult1_03',
 'ind_ahor_fin_ult1_04',
 'ind_ahor_fin_ult1_05',
#  'ind_aval_fin_ult1',
 'ind_aval_fin_ult1_01',
 'ind_aval_fin_ult1_02',
 'ind_aval_fin_ult1_03',
 'ind_aval_fin_ult1_04',
 'ind_aval_fin_ult1_05',
#  'ind_cco_fin_ult1',
 'ind_cco_fin_ult1_01',
 'ind_cco_fin_ult1_02',
 'ind_cco_fin_ult1_03',
 'ind_cco_fin_ult1_04',
 'ind_cco_fin_ult1_05',
 'ind_cco_fin_ult1_back_3',
 'ind_cco_fin_ult1_back_4',
 'ind_cco_fin_ult1_back_5',
#  'ind_cder_fin_ult1',
 'ind_cder_fin_ult1_01',
 'ind_cder_fin_ult1_02',
 'ind_cder_fin_ult1_03',
 'ind_cder_fin_ult1_04',
 'ind_cder_fin_ult1_05',
#  'ind_cno_fin_ult1',
 'ind_cno_fin_ult1_01',
 'ind_cno_fin_ult1_02',
 'ind_cno_fin_ult1_03',
 'ind_cno_fin_ult1_04',
 'ind_cno_fin_ult1_05',
#  'ind_ctju_fin_ult1',
 'ind_ctju_fin_ult1_01',
 'ind_ctju_fin_ult1_02',
 'ind_ctju_fin_ult1_03',
 'ind_ctju_fin_ult1_04',
 'ind_ctju_fin_ult1_05',
 'ind_ctju_fin_ult1_back_3',
 'ind_ctju_fin_ult1_back_4',
 'ind_ctju_fin_ult1_back_5',
#  'ind_ctma_fin_ult1',
 'ind_ctma_fin_ult1_01',
 'ind_ctma_fin_ult1_02',
 'ind_ctma_fin_ult1_03',
 'ind_ctma_fin_ult1_04',
 'ind_ctma_fin_ult1_05',
 'ind_ctma_fin_ult1_back_3',
 'ind_ctma_fin_ult1_back_4',
 'ind_ctma_fin_ult1_back_5',
#  'ind_ctop_fin_ult1',
 'ind_ctop_fin_ult1_01',
 'ind_ctop_fin_ult1_02',
 'ind_ctop_fin_ult1_03',
 'ind_ctop_fin_ult1_04',
 'ind_ctop_fin_ult1_05',
#  'ind_ctpp_fin_ult1',
 'ind_ctpp_fin_ult1_01',
 'ind_ctpp_fin_ult1_02',
 'ind_ctpp_fin_ult1_03',
 'ind_ctpp_fin_ult1_04',
 'ind_ctpp_fin_ult1_05',
#  'ind_deco_fin_ult1',
 'ind_deco_fin_ult1_01',
 'ind_deco_fin_ult1_02',
 'ind_deco_fin_ult1_03',
 'ind_deco_fin_ult1_04',
 'ind_deco_fin_ult1_05',
#  'ind_dela_fin_ult1',
 'ind_dela_fin_ult1_01',
 'ind_dela_fin_ult1_02',
 'ind_dela_fin_ult1_03',
 'ind_dela_fin_ult1_04',
 'ind_dela_fin_ult1_05',
#  'ind_deme_fin_ult1',
 'ind_deme_fin_ult1_01',
 'ind_deme_fin_ult1_02',
 'ind_deme_fin_ult1_03',
 'ind_deme_fin_ult1_04',
 'ind_deme_fin_ult1_05',
#  'ind_ecue_fin_ult1',
 'ind_ecue_fin_ult1_01',
 'ind_ecue_fin_ult1_02',
 'ind_ecue_fin_ult1_03',
 'ind_ecue_fin_ult1_04',
 'ind_ecue_fin_ult1_05',
 'ind_empleado',
 'ind_empleado_01',
 'ind_empleado_02',
 'ind_empleado_03',
 'ind_empleado_04',
 'ind_empleado_05',
#  'ind_fond_fin_ult1',
 'ind_fond_fin_ult1_01',
 'ind_fond_fin_ult1_02',
 'ind_fond_fin_ult1_03',
 'ind_fond_fin_ult1_04',
 'ind_fond_fin_ult1_05',
#  'ind_hip_fin_ult1',
 'ind_hip_fin_ult1_01',
 'ind_hip_fin_ult1_02',
 'ind_hip_fin_ult1_03',
 'ind_hip_fin_ult1_04',
 'ind_hip_fin_ult1_05',
#  'ind_nom_pens_ult1',
 'ind_nom_pens_ult1_01',
 'ind_nom_pens_ult1_02',
 'ind_nom_pens_ult1_03',
 'ind_nom_pens_ult1_04',
 'ind_nom_pens_ult1_05',
 'ind_nom_pens_ult1_back_3',
 'ind_nom_pens_ult1_back_4',
 'ind_nom_pens_ult1_back_5',
#  'ind_nomina_ult1',
 'ind_nomina_ult1_01',
 'ind_nomina_ult1_02',
 'ind_nomina_ult1_03',
 'ind_nomina_ult1_04',
 'ind_nomina_ult1_05',
 'ind_nomina_ult1_back_3',
 'ind_nomina_ult1_back_4',
 'ind_nomina_ult1_back_5',
 'ind_nuevo',
#  'ind_plan_fin_ult1',
 'ind_plan_fin_ult1_01',
 'ind_plan_fin_ult1_02',
 'ind_plan_fin_ult1_03',
 'ind_plan_fin_ult1_04',
 'ind_plan_fin_ult1_05',
#  'ind_pres_fin_ult1',
 'ind_pres_fin_ult1_01',
 'ind_pres_fin_ult1_02',
 'ind_pres_fin_ult1_03',
 'ind_pres_fin_ult1_04',
 'ind_pres_fin_ult1_05',
#  'ind_reca_fin_ult1',
 'ind_reca_fin_ult1_01',
 'ind_reca_fin_ult1_02',
 'ind_reca_fin_ult1_03',
 'ind_reca_fin_ult1_04',
 'ind_reca_fin_ult1_05',
#  'ind_recibo_ult1',
 'ind_recibo_ult1_01',
 'ind_recibo_ult1_02',
 'ind_recibo_ult1_03',
 'ind_recibo_ult1_04',
 'ind_recibo_ult1_05',
 'ind_recibo_ult1_back_3',
 'ind_recibo_ult1_back_4',
 'ind_recibo_ult1_back_5',
#  'ind_tjcr_fin_ult1',
 'ind_tjcr_fin_ult1_01',
 'ind_tjcr_fin_ult1_02',
 'ind_tjcr_fin_ult1_03',
 'ind_tjcr_fin_ult1_04',
 'ind_tjcr_fin_ult1_05',
#  'ind_valo_fin_ult1',
 'ind_valo_fin_ult1_01',
 'ind_valo_fin_ult1_02',
 'ind_valo_fin_ult1_03',
 'ind_valo_fin_ult1_04',
 'ind_valo_fin_ult1_05',
#  'ind_viv_fin_ult1',
 'ind_viv_fin_ult1_01',
 'ind_viv_fin_ult1_02',
 'ind_viv_fin_ult1_03',
 'ind_viv_fin_ult1_04',
 'ind_viv_fin_ult1_05',
 'indext',
 'indfall',
 'indrel',
 'indrel_01',
 'indrel_02',
 'indrel_03',
 'indrel_04',
 'indrel_05',
 'indrel_1mes',
 'indrel_1mes_01',
 'indrel_1mes_02',
 'indrel_1mes_03',
 'indrel_1mes_04',
 'indrel_1mes_05',
 'indresi',
 # 'month',
 # 'ncodpers',
 'nomprov_max',
 'nomprov_mean',
 'nomprov_median',
 'nomprov_min',
 'nomprov_std',
 'num_products_01',
 'num_products_02',
 'num_products_03',
 'num_products_04',
 'num_products_05',
 'pais_residencia_max',
 'pais_residencia_mean',
 'pais_residencia_median',
 'pais_residencia_min',
 'pais_residencia_std',
 'renta',
 'renta61',
 'renta62',
 'renta63',
 'renta64',
 'renta65',
 'renta_01',
 'renta_02',
 'renta_03',
 'renta_04',
 'renta_05',
 'segmento_max',
 'segmento_mean',
 'segmento_median',
 'segmento_min',
 'segmento_std',
 'sexo_max',
 'sexo_mean',
 'sexo_median',
 'sexo_min',
 'sexo_std',
 'tipodom',
 'tiprel_1mes',
 'tiprel_1mes_01',
 'tiprel_1mes_02',
 'tiprel_1mes_03',
 'tiprel_1mes_04',
 'tiprel_1mes_05']


class XgbWrapper(object):
    def __init__(self, seed=2016, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train, seed, x_val, y_val):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.param['seed'] = seed
        dval = xgb.DMatrix(x_val, label=y_val)
        watchlist = [(dtrain, 'train'), (dval, 'val')]
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds, watchlist, early_stopping_rounds=50)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))

nbags = 1


def get_oof(clf):
    pred_oob = np.zeros((X.shape[0], len(le_y.classes_)))
    pred_test = np.zeros((X_test.shape[0], len(le_y.classes_)))

    for i, (train_index, test_index) in enumerate(kf.split(y, y)):
        print "Fold = ", i
        x_tr = X[train_index]
        y_tr = y[train_index]

        x_te = X[test_index]
        y_te = y[test_index]

        pred = np.zeros((x_te.shape[0], len(le_y.classes_)))

        for j in range(nbags):
            x_tr, y_tr = shuffle(x_tr, y_tr, random_state=RANDOM_STATE + i + j)
            clf.train(x_tr, y_tr, RANDOM_STATE + i, x_te, y_te)

            pred += clf.predict(x_te)
            pred_test += clf.predict(X_test)

        pred /= nbags
        pred_oob[test_index] = pred

        score = log_loss(y_te, pred)
        print('Fold ', i, '- LogLoss:', score)

    return pred_oob, pred_test


train, test = clean_data.label_encoded(cached=False)

train = train.fillna(-99999)
test = test.fillna(-99999)

# Classes with too few elements excluded
train = train[~train['added_product'].isin(['ind_cder_fin_ult1',
                                            'ind_pres_fin_ult1',
                                            'ind_hip_fin_ult1',
                                            'ind_viv_fin_ult1'])]

X = train[features].values
X_test = test[features].values
#
# X_lr = pd.read_csv('oof/lr_train_1.csv').drop('ncodpers', 1).values
# X_test_lr = pd.read_csv('oof/lr_test_1.csv').drop('ncodpers', 1).values
#
# X = np.hstack([X, X_lr])
# X_test = np.hstack([X_test, X_test_lr])

print X.shape, X_test.shape

print train['added_product'].value_counts()

le_y = LabelEncoder()
y = le_y.fit_transform(train['added_product'])

n_folds = 5
RANDOM_STATE = 2016

num_train = X.shape[0]
num_test = X_test.shape[0]

kf = StratifiedKFold(n_folds, shuffle=True, random_state=RANDOM_STATE)

xgb_params = {
    'min_child_weight': 10,
    'eta': 0.1,
    'colsample_bytree': 0.7,
    'max_depth': 6,
    'subsample': 0.9,
    # 'max_delta_step': 0.01,
    # 'alpha': 5,
    'lambda': 5,
    'gamma': 0,
    'silent': 1,
    # 'base_score': y_t.mean(),
    'verbose_eval': 1,
    'seed': RANDOM_STATE,
    'nrounds': 10000,
    'objective': "multi:softprob",
    'num_class': len(le_y.classes_),
    'eval_metric': ['merror', 'mlogloss']
}


xg = XgbWrapper(seed=RANDOM_STATE, params=xgb_params)
xg_oof_train, xg_oof_test = get_oof(xg)

print xg_oof_train.shape
print set(y), len(y)
print("XG-CV: {}".format(log_loss(y, xg_oof_train)))

print '[{datetime}] Saving train probs'.format(datetime=str(datetime.datetime.now()))
oof_train = pd.DataFrame(xg_oof_train, columns=le_y.classes_)
oof_train['ncodpers'] = train['ncodpers'].values
oof_train.to_csv('oof/xgb_train_1.csv', index=False)

print '[{datetime}] Saving test probs'.format(datetime=str(datetime.datetime.now()))
xg_oof_test /= (n_folds * nbags)

oof_test = pd.DataFrame(xg_oof_test, columns=le_y.classes_)
oof_test['ncodpers'] = test['ncodpers'].values
oof_test.to_csv('oof/xgb_test_1.csv', index=False)

print '[{datetime}] Saving submission probs'.format(datetime=str(datetime.datetime.now()))
submission = clean_data.submission(oof_test, test)
submission.to_csv('submissions/xgb_sub_1.csv', index=False)
