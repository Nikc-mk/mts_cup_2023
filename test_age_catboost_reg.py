import warnings

warnings.filterwarnings('ignore')
import pyarrow.parquet as pq
import pandas as pd
import bisect
import sklearn.metrics as m
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

SPLIT_SEED = 42
DATA_FILE = 'data_out/data_for_learn_parquet_last'
TARGET_FILE = 'data_in/public_train.pqt'

data = pq.read_table(f"{DATA_FILE}").to_pandas()
data = data.fillna(0)

data['avg_request_cnt/stddev_request_cnt'] = data['avg_request_cnt'] / data['stddev_request_cnt']
data['avg_night_request_cnt/stddev_night_request_cnt'] = data['avg_night_request_cnt'] / data[
    'stddev_night_request_cnt']
data['avg_day_request_cnt/stddev_day_request_cnt'] = data['avg_day_request_cnt'] / data['stddev_day_request_cnt']
data['avg_morning_request_cnt/stddev_morning_request_cnt'] = data['avg_morning_request_cnt'] / data[
    'stddev_morning_request_cnt']
data['avg_evening_request_cnt/stddev_evening_request_cnt'] = data['avg_evening_request_cnt'] / data[
    'stddev_evening_request_cnt']

data['avg_sum_date_request_cnt/stddev_sum_date_request_cnt'] = data['avg_sum_date_request_cnt'] / data[
    'stddev_sum_date_request_cnt']
data['day_avg_sum_date_request_cnt/day_stddev_sum_date_request_cnt'] = data['day_avg_sum_date_request_cnt'] / data[
    'day_stddev_sum_date_request_cnt']
data['night_avg_sum_date_request_cnt/night_stddev_sum_date_request_cnt'] = data['night_avg_sum_date_request_cnt'] / \
                                                                           data['night_stddev_sum_date_request_cnt']
data['morning_avg_sum_date_request_cnt/morning_stddev_sum_date_request_cnt'] = data[
                                                                                   'morning_avg_sum_date_request_cnt'] / \
                                                                               data[
                                                                                   'morning_stddev_sum_date_request_cnt']
data['evening_avg_sum_date_request_cnt/evening_stddev_sum_date_request_cnt'] = data[
                                                                                   'evening_avg_sum_date_request_cnt'] / \
                                                                               data[
                                                                                   'evening_stddev_sum_date_request_cnt']

data['day_avg_sum_date_request_cnt/avg_sum_date_request_cnt'] = data['day_avg_sum_date_request_cnt'] / data[
    'avg_sum_date_request_cnt']
data['night_avg_sum_date_request_cnt/avg_sum_date_request_cnt'] = data['night_avg_sum_date_request_cnt'] / data[
    'avg_sum_date_request_cnt']
data['morning_avg_sum_date_request_cnt/avg_sum_date_request_cnt'] = data['morning_avg_sum_date_request_cnt'] / data[
    'avg_sum_date_request_cnt']
data['evening_avg_sum_date_request_cnt/avg_sum_date_request_cnt'] = data['evening_avg_sum_date_request_cnt'] / data[
    'avg_sum_date_request_cnt']

all_usr_emb = pq.read_table(f"data_in/all_usr_emb_f80_i40.parquet").to_pandas()
data = data.merge(all_usr_emb, how='left', on=['user_id'])

target = pq.read_table(f"{TARGET_FILE}").to_pandas()
data_t_age = target.merge(data, how='left', on=['user_id'])

data_t_age['age'] = data_t_age['age'].map(str)
data_t_age = data_t_age[data_t_age['age'] != 'None']
data_t_age = data_t_age[data_t_age['age'] != 'NA']
data_t_age['age'] = data_t_age['age'].map(float)
data_t_age = data_t_age[data_t_age['age'] >= 18]
print(data_t_age['age'].count())

x_train, x_test, y_train, y_test = train_test_split(
    data_t_age.drop(['user_id', 'age', 'is_male'], axis=1), data_t_age['age'], test_size=0.25, random_state=SPLIT_SEED)

clf_age = CatBoostRegressor(thread_count=3, iterations=2000)
clf_age.fit(x_train, y_train, verbose=False)


def age_bucket(x):
    return bisect.bisect_right([18, 25, 35, 45, 55, 65], x)


predict_age = pd.DataFrame()
predict_age['age'] = clf_age.predict(x_test)
predict_age['age'] = predict_age['age'].map(age_bucket)
y_test = pd.DataFrame(data=y_test)
y_test['age'] = y_test['age'].map(age_bucket)
predict_age['age'][predict_age['age'] == 0] = predict_age['age'] + 1

print(predict_age['age'].value_counts())
print(y_test['age'].value_counts())
print(predict_age['age'].count())
print(y_test['age'].count())

print(m.classification_report(y_test['age'], predict_age['age'],
                              target_names=['18-25', '25-34', '35-44', '45-54', '55-65', '65+']))
