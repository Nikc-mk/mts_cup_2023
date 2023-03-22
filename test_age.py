import warnings

warnings.filterwarnings('ignore')
import pyarrow.parquet as pq
import bisect
import sklearn.metrics as m
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

SPLIT_SEED = 42
DATA_FILE = 'data_in/all_usr_emb_f80_i40.parquet'
TARGET_FILE = 'data_in/public_train.pqt'

data = pq.read_table(f"{DATA_FILE}").to_pandas()

data_1 = pq.read_table(f"data_in/count_usr_emb_f50_i30.parquet").to_pandas()
data = data.merge(data_1, how='left', on=['user_id'])
data_2 = pq.read_table(f"data_in/night_bas_usr_emb_f20_i50.parquet").to_pandas()
data = data.merge(data_2, how='left', on=['user_id'])

data = data.fillna(0)


# all_usr_emb = pq.read_table(f"data_in/all_usr_emb_f80_i40.parquet").to_pandas()
# data = data.merge(all_usr_emb, how='left', on=['user_id'])


def age_bucket(x):
    return bisect.bisect_right([18, 25, 35, 45, 55, 65], x)


target = pq.read_table(f"{TARGET_FILE}").to_pandas()
data_t_age = target.merge(data, how='left', on=['user_id'])

data_t_age['age'] = data_t_age['age'].map(str)
data_t_age = data_t_age[data_t_age['age'] != 'None']
data_t_age = data_t_age[data_t_age['age'] != 'NA']
data_t_age['age'] = data_t_age['age'].map(float)
data_t_age['age'] = data_t_age['age'].map(age_bucket)
data_t_age = data_t_age[data_t_age['age'] != 0]
print(data_t_age['age'].value_counts())

x_train, x_test, y_train, y_test = train_test_split(
    data_t_age.drop(['user_id', 'age', 'is_male'], axis=1), data_t_age['age'], test_size=0.25, random_state=SPLIT_SEED)

clf_age = CatBoostClassifier(thread_count=3
                             , iterations=1000
                             , random_seed=SPLIT_SEED, learning_rate=0.05, early_stopping_rounds=100
                             , eval_metric='TotalF1'
                             # , loss_function='MultiClass'
                             , classes_count=6
                             , class_names=[1, 2, 3, 4, 5, 6]
                             )
clf_age.fit(x_train, y_train, verbose=False, eval_set=(x_test, y_test))
print(m.classification_report(y_test, clf_age.predict(x_test),
                              target_names=['18-25', '25-34', '35-44', '45-54', '55-65', '65+']))



# clf_age.get_feature_importance(prettified=True).to_csv(f'data_out/clf_age_importance.csv', index=False, mode='w')

#     accuracy                           0.44     67413
#    macro avg       0.45      0.30      0.31     67413
# weighted avg       0.44      0.44      0.41     67413