import psycopg2
import pandas as pd
from datetime import datetime
from datetime import date
import os
import pyprind
import numpy as np
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import datetime as dt
import matplotlib.pyplot as plt

class rawdata:
    def __init__(self, table):
        self.rawdata = pd.DataFrame()
        self.connect_str = "dbname='******' user='******' host='******' password='******'"
        self.table = table
        self.column_names = []
    def fetch(self): 
        try:
            # use our connection values to establish a connection
            conn = psycopg2.connect(self.connect_str)
            # create a psycopg2 cursor that can execute queries
            cursor = conn.cursor()
            # create a new table with a single column called "name"
            # run a SELECT statement - no data in there, but we can try it
            cursor.execute("""SELECT * from {0}""".format(self.table))
            self.column_names = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            # fetch column names from table
            #close session
            conn.commit()
            cursor.close()
            #combine data into pandas
            self.rawdata = pd.DataFrame.from_records(rows, columns=self.column_names)
        except Exception as e:
            print("Uh oh, can't connect. Invalid dbname, user or password?")
            print(e)

call_log = rawdata("five9.call_log")
call_log.fetch()
call_log_copy = call_log.rawdata.copy()
# call_log_copy['calldate'] = call_log_copy.timestamp.map(lambda x: x.date())


connect_str = "dbname='reporting' user='gfangyuan' host='reporting.ckpglb17yttu.us-east-1.rds.amazonaws.com' password='chimney'"
conn = psycopg2.connect(connect_str)
name_c_com = pd.read_sql_query("""select mail_date, count(1) mail_nums from directmail.clr_remail
group by mail_date 
order by mail_date asc;""", conn)
conn.close()

# directmail = rawdata("directmail.clr_remail")
# directmail.fetch()
# directmail_copy = directmail.rawdata.copy()
# # directmail_copy['sxn'] = directmail_copy.sxn.astype(str)

del call_log#, directmail

call_log_copy = call_log_copy[call_log_copy.calltype == 'Inbound']
call_log_copy['time_by_day'] = call_log_copy.timestamp.map(lambda x: x.date())


day_t = call_log_copy.groupby('time_by_day').size().reset_index()
day_t.columns = ['time_by_day', 'call_nums']
df = day_t.copy()
# while (df.iloc[-1]['time_by_day'] + timedelta(days=14) < datetime(2017,12,31).date()):
#     df = df.append({'time_by_day': df.iloc[-1]['time_by_day']+timedelta(days=7), 'call_nums': 380370}, ignore_index=True)
for i in range(7):
    df = df.append({'time_by_day': day_t.iloc[-1]['time_by_day']+timedelta(days=i+1), 'call_nums': 0}, ignore_index=True)

day_t = df.copy()

# name_c_com = directmail_copy.groupby(['mail_date']).size().reset_index()
# name_c_com.columns = ['mail_date','mail_nums']
dff = name_c_com.copy()
while (dff.iloc[-1]['mail_date'] < datetime(2017,12,31).date()):
    dff = dff.append({'mail_date': dff.iloc[-1]['mail_date']+timedelta(days=7), 'mail_nums': 380370}, ignore_index=True)

name_c_com = dff.copy()



def date_digitize(x):
    for i in range(len(name_c_com.mail_date)-1):
        if (x > name_c_com.mail_date[i]) & (x <= name_c_com.mail_date[i+1]):
            return i
    if x > name_c_com.mail_date[len(name_c_com.mail_date)-1]:
        return len(name_c_com.mail_date) - 1


day_t['mail_volume_1_most_recent_prior'] = day_t.time_by_day.map(lambda x: date_digitize(x))
day_t.loc[80, 'call_nums'] = 3177
day_t.loc[142, 'call_nums'] = 3744

for i in range(8):
    col = 'mail_volume_'+str(i+2)+'_most_recent_prior'
    day_t[col] = day_t['mail_volume_1_most_recent_prior'].map(lambda x: x-i-1 if x-i-1 >= 0 else np.nan)

col1 = 'mail_volume_1_most_recent_prior'
col = 'days_from_last_1_maildate'
day_t[col] = day_t.apply(lambda x: (x.time_by_day - name_c_com.loc[int(x[col1]), 'mail_date']).days if not np.isnan(x[col1]) else np.nan, axis = 1)

for i in range(8):
    col1 = 'mail_volume_'+str(i+2)+'_most_recent_prior'
    col = 'days_from_last_'+str(i+2)+'_maildate'
    day_t[col] = day_t.apply(lambda x: (x.time_by_day - name_c_com.loc[int(x[col1]), 'mail_date']).days if not np.isnan(x[col1]) else np.nan, axis = 1)

col = 'mail_volume_1_most_recent_prior'
day_t[col] = day_t[col].map(lambda x: name_c_com.loc[int(x), 'mail_nums'] if not np.isnan(x) else np.nan)

for i in range(8):
    col = 'mail_volume_'+str(i+2)+'_most_recent_prior'
    day_t[col] = day_t[col].map(lambda x: name_c_com.loc[int(x), 'mail_nums'] if not np.isnan(x) else np.nan)



day_t['weekday'] = day_t.time_by_day.map(lambda x: x.weekday()+1)
day_t['weekday_square'] = day_t.weekday.map(lambda x: x*x)
day_t['weekend_or_not'] = day_t.time_by_day.map(lambda x: True if x.weekday() > 4 else False)

day_t['call_number_lag_6'] = day_t['call_nums'].shift(6)
day_t['call_number_lag_7'] = day_t['call_nums'].shift(7)
day_t['call_number_lag_8'] = day_t['call_nums'].shift(8)
day_t['call_number_lag_9'] = day_t['call_nums'].shift(9)

# add absolute week time interval and pick the most recent direct mail days, if equal at both ends, pick the oldest time
# def absolute_date_digitize(x, st, et):
#     if name_c_com.mail_nums[((x - name_c_com.mail_date) >= timedelta(days=st)) & ((x - name_c_com.mail_date) <= timedelta(days=et))].empty:
#         return np.nan
#     else:
#         return name_c_com.mail_nums[((x - name_c_com.mail_date) >= timedelta(days=st)) & ((x - name_c_com.mail_date) <= timedelta(days=et))].reset_index(drop = True)[0]


# day_t['abs_mail_date_back1421'] = day_t.time_by_day.map(lambda x: absolute_date_digitize(x, 14, 21))
# day_t['abs_mail_date_back2128'] = day_t.time_by_day.map(lambda x: absolute_date_digitize(x, 21, 28))

# def gm(arr):
#     df = pd.DataFrame(arr)
#     v = df.cumsum()
#     return v.iloc[-2]

# temp = day_t[['time_by_day','weekday', 'call_nums']]
# tt = pd.rolling_apply(temp, 5 ,lambda x: gm(x))    

# day_t['last_weeks_data_1'] = day_t.groupby(['weekday']).rolling_apply(temp, 5 ,lambda x: gm(x))
#     reset_index(drop=True)

temp_1_week_interval = day_t.groupby(['weekday'])[['time_by_day','call_nums']].rolling(5).apply(lambda x: x[0]).reset_index()
temp_2_week_interval = day_t.groupby(['weekday'])[['time_by_day','call_nums']].rolling(5).apply(lambda x: x[1]).reset_index()
temp_3_week_interval = day_t.groupby(['weekday'])[['time_by_day','call_nums']].rolling(5).apply(lambda x: x[2]).reset_index()
temp_4_week_interval = day_t.groupby(['weekday'])[['time_by_day','call_nums']].rolling(5).apply(lambda x: x[3]).reset_index()

del temp_1_week_interval['level_1'], temp_2_week_interval['level_1'], temp_3_week_interval['level_1'], temp_4_week_interval['level_1']
del temp_2_week_interval['weekday'], temp_3_week_interval['weekday'], temp_4_week_interval['weekday']

total_temp = temp_1_week_interval.merge(temp_2_week_interval, on='time_by_day').merge(temp_3_week_interval, on='time_by_day').merge(temp_4_week_interval, on = 'time_by_day').sort_values(['time_by_day'])
total_temp.columns = ['weekday', 'time_by_day', 'call_volume_4_wk_prior', 'call_volume_3_wk_prior', 'call_volume_2_wk_prior', 'call_volume_1_wk_prior']

day_t = pd.merge(day_t, total_temp[['time_by_day', 'call_volume_4_wk_prior', 'call_volume_3_wk_prior', 'call_volume_2_wk_prior', 'call_volume_1_wk_prior']], on = ['time_by_day'])
day_t['avg_4_weeks'] = day_t[['time_by_day', 'call_volume_4_wk_prior', 'call_volume_3_wk_prior', 'call_volume_2_wk_prior', 'call_volume_1_wk_prior']].mean(1)
# take all weekday == 1 and pick the last 4 elements
# for 7/4, 9/4, just replace it manually

day_t.to_csv("Desktop/direct mail analysis/new/data_with_predict.csv")



data_waited_to_predict = day_t[day_t.time_by_day >= datetime(2017, 11, 4).date()]

data_waited_to_predict['prediction'] = data_waited_to_predict.apply(lambda x: predict(x), axis = 1)


new_day_t = data_waited_to_predict[['time_by_day', 'prediction']]


df_head = df[df.time_by_day <= datetime(2017, 11,12).date()]
df_tail = new_day_t[new_day_t.time_by_day > datetime(2017, 11,12).date()]
df_tail.columns = df_head.columns
df_into_loop = pd.concat([df_head, df_tail], axis = 0)

df_into_loop1 = outgrow_by_7(df_into_loop)
df_into_loop2 = outgrow_by_7(df_into_loop1)
df_into_loop3 = outgrow_by_7(df_into_loop2)
df_into_loop4 = outgrow_by_7(df_into_loop3)
df_into_loop5 = outgrow_by_7(df_into_loop4)
df_into_loop6 = outgrow_by_7(df_into_loop5)

df_into_loop6.to_csv("Desktop/direct mail analysis/new/data_predicted_to_the_year_end.csv")

import numpy as np
import pandas as pd
import time
import re
from datetime import datetime
import sys
import os
import re
import calendar

def predict(row):
    round_days_from_last_2_maildate = np.float32(row[u'days_from_last_2_maildate'])
    round_call_number_lag_7 = np.float32(row[u'call_number_lag_7'])
    round_weekend_or_not = np.float32(row[u'weekend_or_not'])
    round_weekday = np.float32(row[u'weekday'])
    round_weekday_square = np.float32(row[u'weekday_square'])
    round_4_week_interval = np.float32(row[u'call_volume_4_wk_prior'])
    round_days_from_last_3_maildate = np.float32(row[u'days_from_last_3_maildate'])
    round_days_from_last_6_maildate = np.float32(row[u'days_from_last_6_maildate'])
    round_3_week_interval = np.float32(row[u'call_volume_3_wk_prior'])
    round_2_week_interval = np.float32(row[u'call_volume_2_wk_prior'])
    round_mail_date_seg_6 = np.float32(row[u'mail_volume_6_most_recent_prior'])
    round_mail_date_seg_5 = np.float32(row[u'mail_volume_5_most_recent_prior'])
    round_mail_date_seg_4 = np.float32(row[u'mail_volume_4_most_recent_prior'])
    round_mail_date_seg_3 = np.float32(row[u'mail_volume_3_most_recent_prior'])
    round_days_from_last_4_maildate = np.float32(row[u'days_from_last_4_maildate'])
    round_avg_4_weeks = np.float32(row[u'avg_4_weeks'])
    return sum([
        2618.5006813,
        -9.4507029569836889689E-05 * (round_mail_date_seg_5),
        -0.00037947952152292023032 * (round_mail_date_seg_6),
           -0.36521587215431572382 * (round_days_from_last_6_maildate),
            -18.290128684684439975 * (round_weekday),
            -6.1080595947197613427 * (round_weekday_square),
             -657.5691429629562208 * (round_weekend_or_not),
           0.088367020683345912091 * (round_4_week_interval),
           0.067006394254979464997 * (round_3_week_interval),
           0.035259847673089164677 * (round_2_week_interval),
           0.025412826760337386273 * (round_avg_4_weeks),
            -99.686931323058203702 * (round_weekday > 5.5 and 
                                     round_2_week_interval <= 992.5),
            -33.231257316061466156 * (round_weekday_square > 42.5 and 
                                     round_weekend_or_not > 0.5),
             48.855948795466062506 * (round_mail_date_seg_6 <= 604451.5 and 
                                     round_days_from_last_2_maildate <= 11.5 and 
                                     round_weekday <= 5.5 and 
                                     round_3_week_interval > 3025.5),
             -130.3029506618082678 * (round_4_week_interval <= 1413.5),
             140.13401538589153006 * (round_mail_date_seg_5 <= 254943.5 and 
                                     round_mail_date_seg_6 <= 275895.5 and 
                                     round_call_number_lag_7 > 1422.0),
             3.7967387857497723047 * (round_mail_date_seg_3 > 449096.5 and 
                                     round_call_number_lag_7 > 1422.0 and 
                                     round_3_week_interval > 3605.5),
             39.838313546288667055 * (round_mail_date_seg_5 <= 700645.0 and 
                                     round_days_from_last_3_maildate <= 18.5 and 
                                     round_4_week_interval > 3255.5 and 
                                     round_2_week_interval > 1260.5),
            -80.435115397640899459 * (round_mail_date_seg_3 <= 580645.5 and 
                                     round_mail_date_seg_6 > 275895.5 and 
                                     round_call_number_lag_7 <= 1473.0),
            -10.287562958360629395 * (round_mail_date_seg_6 > 275895.5 and 
                                     round_3_week_interval <= 1595.5),
             45.058487845321529619 * (round_mail_date_seg_4 > 254943.5 and 
                                     round_mail_date_seg_5 > 205820.0 and 
                                     round_mail_date_seg_6 <= 604451.5 and 
                                     round_2_week_interval > 1260.5),
            -103.75097910003796642 * (round_call_number_lag_7 <= 2784.0 and 
                                     round_3_week_interval <= 2696.5 and 
                                     round_avg_4_weeks <= 3695.75),
             27.487661681806741854 * (round_mail_date_seg_6 <= 275895.5 and 
                                     round_call_number_lag_7 > 3052.0 and 
                                     round_3_week_interval > 3003.0 and 
                                     round_avg_4_weeks <= 3695.75),
             -14.01653799310573234 * (round_mail_date_seg_6 > 234075.0 and 
                                     round_weekday_square > 6.5 and 
                                     round_2_week_interval <= 3278.0 and 
                                     round_avg_4_weeks <= 3695.75),
             -69.95632907491243202 * (round_mail_date_seg_6 > 254943.5 and 
                                     round_days_from_last_2_maildate > 5.5 and 
                                     round_call_number_lag_7 <= 3856.0 and 
                                     round_3_week_interval <= 2772.5),
             20.473507129906685975 * (round_days_from_last_6_maildate <= 38.5 and 
                                     round_call_number_lag_7 > 3038.0 and 
                                     round_3_week_interval > 2982.0),
              -35.5701741934210105 * (round_days_from_last_4_maildate <= 24.5 and 
                                     round_call_number_lag_7 <= 3665.5 and 
                                     round_2_week_interval <= 3239.0),
             114.22684097339987375 * (round_mail_date_seg_6 <= 604451.5 and 
                                     round_days_from_last_4_maildate <= 26.5 and 
                                     round_2_week_interval > 1260.5),
             -132.3843863766922766 * (round_call_number_lag_7 <= 3289.5 and 
                                     round_3_week_interval <= 3003.0 and 
                                     round_2_week_interval <= 3417.0 and 
                                     round_avg_4_weeks <= 3380.75),
            -8.5520550587373449503 * (round_mail_date_seg_3 <= 425324.0 and 
                                     round_days_from_last_2_maildate > 7.5 and 
                                     round_4_week_interval <= 2929.5),
             13.022547872498288157 * (round_mail_date_seg_4 > 205820.0 and 
                                     round_mail_date_seg_6 <= 604451.5 and 
                                     round_days_from_last_2_maildate > 5.5 and 
                                     round_2_week_interval > 791.0),
             60.230429272540270347 * (round_call_number_lag_7 > 1307.5 and 
                                     round_3_week_interval > 711.0 and 
                                     round_2_week_interval > 2839.0)    ])




