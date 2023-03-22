import warnings
warnings.filterwarnings('ignore')
import pyarrow.parquet as pq
import bisect
import sklearn.metrics as m
from xgboost import XGBClassifier, DMatrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SPLIT_SEED = 42
DATA_FILE = 'data_out/data_for_learn_parquet_last'
TARGET_FILE = 'data_in/public_train.pqt'

data = pq.read_table(f"{DATA_FILE}").to_pandas()
data = data.fillna(0)


all_usr_emb = pq.read_table(f"data_in/all_usr_emb_f80_i40.parquet").to_pandas()
data = data.merge(all_usr_emb, how = 'left', on = ['user_id'])
data_usr_emb =pq.read_table(f"data_in/data_usr_emb_f50_i40.parquet").to_pandas()
data = data.merge(data_usr_emb, how = 'left', on = ['user_id'])
target = pq.read_table(f"{TARGET_FILE}").to_pandas()
data_t_is_male = target.merge(data, how = 'left', on = ['user_id'])

data_t_is_male['is_male'] = data_t_is_male['is_male'].map(str)
data_t_is_male = data_t_is_male[data_t_is_male['is_male'] != 'None']
data_t_is_male = data_t_is_male[data_t_is_male['is_male'] != 'NA']
data_t_is_male['is_male'] = data_t_is_male['is_male'].map(int)
print(data_t_is_male['is_male'].value_counts())


x_train, x_test, y_train, y_test = train_test_split(\
    data_t_is_male.drop(['user_id', 'age', 'is_male'], axis = 1), data_t_is_male['is_male'], test_size = 0.25, random_state = SPLIT_SEED)

# xgb_scala = StandardScaler()
# x_train_scaled = xgb_scala.fit_transform(x_train)
# x_test_scaled = xgb_scala.transform(x_test)

clf_is_male = XGBClassifier(verbosity=0, nthread=3, objective="binary:logistic")
clf_is_male.fit(x_train, y_train, verbose = 0)
print(f'GINI по полу {2 * m.roc_auc_score(y_test, clf_is_male.predict(x_test)) - 1:2.3f}')

print(clf_is_male.predict(x_test))


# 0.677