import warnings
warnings.filterwarnings('ignore')
import pyarrow.parquet as pq
import bisect
import sklearn.metrics as m
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split



SPLIT_SEED = 42
DATA_FILE = 'data_out/data_for_learn_parquet_last'
TARGET_FILE = 'data_in/public_train.pqt'

data = pq.read_table(f"{DATA_FILE}").to_pandas()
data = data.fillna(0)

data['avg_request_cnt/stddev_request_cnt'] = data['avg_request_cnt'] / data['stddev_request_cnt']
data['avg_night_request_cnt/stddev_night_request_cnt'] = data['avg_night_request_cnt'] / data['stddev_night_request_cnt']
data['avg_day_request_cnt/stddev_day_request_cnt'] = data['avg_day_request_cnt'] / data['stddev_day_request_cnt']
data['avg_morning_request_cnt/stddev_morning_request_cnt'] = data['avg_morning_request_cnt'] / data['stddev_morning_request_cnt']
data['avg_evening_request_cnt/stddev_evening_request_cnt'] = data['avg_evening_request_cnt'] / data['stddev_evening_request_cnt']

data['avg_sum_date_request_cnt/stddev_sum_date_request_cnt'] = data['avg_sum_date_request_cnt'] / data['stddev_sum_date_request_cnt']
data['day_avg_sum_date_request_cnt/day_stddev_sum_date_request_cnt'] = data['day_avg_sum_date_request_cnt'] / data['day_stddev_sum_date_request_cnt']
data['night_avg_sum_date_request_cnt/night_stddev_sum_date_request_cnt'] = data['night_avg_sum_date_request_cnt'] / data['night_stddev_sum_date_request_cnt']
data['morning_avg_sum_date_request_cnt/morning_stddev_sum_date_request_cnt'] = data['morning_avg_sum_date_request_cnt'] / data['morning_stddev_sum_date_request_cnt']
data['evening_avg_sum_date_request_cnt/evening_stddev_sum_date_request_cnt'] = data['evening_avg_sum_date_request_cnt'] / data['evening_stddev_sum_date_request_cnt']

data['day_avg_sum_date_request_cnt/avg_sum_date_request_cnt'] = data['day_avg_sum_date_request_cnt'] / data['avg_sum_date_request_cnt']
data['night_avg_sum_date_request_cnt/avg_sum_date_request_cnt'] = data['night_avg_sum_date_request_cnt'] / data['avg_sum_date_request_cnt']
data['morning_avg_sum_date_request_cnt/avg_sum_date_request_cnt'] = data['morning_avg_sum_date_request_cnt'] / data['avg_sum_date_request_cnt']
data['evening_avg_sum_date_request_cnt/avg_sum_date_request_cnt'] = data['evening_avg_sum_date_request_cnt'] / data['avg_sum_date_request_cnt']


all_usr_emb = pq.read_table(f"data_in/all_usr_emb_f80_i40.parquet").to_pandas()
data = data.merge(all_usr_emb, how = 'left', on = ['user_id'])
target = pq.read_table(f"{TARGET_FILE}").to_pandas()
data_t_is_male = target.merge(data, how = 'left', on = ['user_id'])

data_t_is_male['is_male'] = data_t_is_male['is_male'].map(str)
data_t_is_male = data_t_is_male[data_t_is_male['is_male'] != 'None']
data_t_is_male = data_t_is_male[data_t_is_male['is_male'] != 'NA']
data_t_is_male['is_male'] = data_t_is_male['is_male'].map(int)
print(data_t_is_male['is_male'].value_counts())


x_train, x_test, y_train, y_test = train_test_split(\
    data_t_is_male.drop(['user_id', 'age', 'is_male'], axis = 1), data_t_is_male['is_male'], test_size = 0.25, random_state = SPLIT_SEED)

clf_is_male = LGBMClassifier(n_jobs=4, objective='binary', num_iterations=2000
                             , metric='binary_logloss')
clf_is_male.fit(x_train, y_train, verbose = False)
print(f'GINI по полу {2 * m.roc_auc_score(y_test, clf_is_male.predict(x_test)) - 1:2.3f}')

# clf_is_male.feature_importances_(prettified=True).to_csv(f'data_out/clf_lightgbm_is_male_importance.csv', index = False, mode='w')
# GINI по полу 0.449