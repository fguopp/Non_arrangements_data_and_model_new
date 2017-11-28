import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
from datetime import date
import os
import pyprind
from datetime import timedelta
from dateutil.relativedelta import relativedelta

connect_str = "dbname='reporting' user='*********' host='******' password='*******'"
conn = psycopg2.connect(connect_str)
call_log_copy = pd.read_sql_query("""select timestamp::date, count(1) from five9.call_log
where calltype='Inbound'
and skill != 'IB Arrangements'
group by 1 order by 1;
""", conn)
application_copy = pd.read_sql_query("""select
a."Date"
,sum(a."Applications")
from tableau_reporting.daily_marketing_applications a
group by 1
order by 1;
""", conn)
conn.close()

application_copy.columns = call_log_copy.columns
df = pd.merge(application_copy, call_log_copy, on = 'timestamp')
# df.to_csv("Desktop/call_log_and_application analysis/df.csv")

df_1 = df[df.timestamp >= datetime(2017,2,19).date()].reset_index(drop=True)
df_1.columns = ['timestamp', 'application_num', 'call_log_num']
df_1['weekday'] = df_1.timestamp.map(lambda x: x.weekday())

df_1 = df_1[df_1.weekday <= 4]
# df_add = df_1[df_1.timestamp == datetime(2017,4,11).date()]
# df_add = pd.DataFrame({})



# df_c = pd.read_csv("Desktop/call_log_and_application analysis/data_cleaned.csv")
# t = df_c.copy()
# t.index = t.timestamp
# del t['timestamp']

# t.index = pd.to_datetime(t.index, infer_datetime_format=True)
# a = t.application_copy.resample('W-SAT').sum()
# b = a.resample('D').last().bfill()
# ff = pd.concat([t, b], axis = 1).bfill()
# ff['weekday'] = ff.index.map(lambda x: x.weekday())
# ff['weekday2'] = ff.weekday.map(lambda x: x*x)
# ff.columns = ['application', 'call_log', 'weekday', 'application_week', 'weekday2']
# for i in range(3):
# 	col = 'application_delay_'+ str(i+1) + '_week'
# 	ff[col] = ff.application_week.shift(7*(i+1))

# ff['weekend_or_not'] = ff.index.map(lambda x: True if x.weekday() > 4 else False)
# # ff.loc[80, 'call_log'] = 3177
# # ff.loc[142, 'call_log'] = 3744
# ff.loc[pd.Timestamp('2017-07-04'),'call_log'] = 3177
# ff.loc[pd.Timestamp('2017-09-04'),'call_log'] = 3744


# for i in range(6,10):
# 	col1 = 'application_delay_'+ str(i+1) + '_day'
# 	ff[col1] = ff.application.shift(i+1)

# for i in range(6, 10):
# 	col1 = 'call_log_delay_'+ str(i+1) + '_day'
# 	ff[col1] = ff.call_log.shift(i+1)

# for i in range(4):
# 	col = 'call_log_prior_'+ str(i+1) + '_week'
# 	ff[col] = ff.call_log.shift(7*(i+1))

# ff['avg_4_weeks'] = ff[['call_log_prior_1_week', 'call_log_prior_2_week', 'call_log_prior_3_week', 'call_log_prior_4_week']].mean(1)
# ff.to_csv("Desktop/call_log_and_application analysis/ff.csv")

# code to generate future data for 1 week ahead
# t_pseduo = [[datetime(2017,4,12).date(), 3600.5, 3829, 4],\
# 	[datetime(2017,4,13).date(), 3875.5, 2982, 5], \
# 	[datetime(2017,4,14).date(), 3306.5, 2449, 6]]
t_pseduo = list()

projection = pd.read_csv("Desktop/call_log_and_application analysis/projections.csv")
projection['timestamp'] = pd.to_datetime(projection.timestamp, format='%m/%d/%y')
sliced_projection = projection[(projection.timestamp <= datetime.today().date()+timedelta(days=20)) & (projection.timestamp >= datetime(2017, 11, 21))]

for i in range(14):
	apps = sliced_projection[(sliced_projection.timestamp == max(df_1.timestamp)+timedelta(days=i+1))]['application_num'].iloc[0]
	t_pseduo.append([max(df_1.timestamp)+timedelta(days=i+1), apps, np.nan, np.nan])

t_pseduo = pd.DataFrame(t_pseduo, columns = df_1.columns)
p_df = df_1.append(t_pseduo).reset_index(drop = True).sort_values(['timestamp'])
p_df.at[156, 'application_num'] = sliced_projection[(sliced_projection.timestamp == datetime(2017,11,21))]['application_num'].iloc[0]

p_dft = p_df.copy()
p_dft['weekday'] = p_dft.timestamp.map(lambda x: x.weekday())
p_dft = p_dft[p_dft.weekday <= 4]

p_dft.index = p_dft.timestamp
del p_dft['timestamp']

p_dft.index = pd.to_datetime(p_dft.index, infer_datetime_format=True)
a = p_dft.application_num.resample('W-SAT').sum()
b = a.resample('D').last().bfill()

pdf_ff = pd.concat([p_dft, b], axis = 1).bfill()
pdf_ff['weekday'] = pdf_ff.index.map(lambda x: x.weekday())
pdf_ff = pdf_ff[pdf_ff.weekday <= 4]

# pdf_ff['weekday2'] = pdf_ff.weekday.map(lambda x: x*x)
pdf_ff.columns = ['application', 'call_log', 'weekday', 'application_week']
for i in range(3):
	col = 'application_delay_'+ str(i+1) + '_week'
	pdf_ff[col] = pdf_ff.application_week.shift(5*(i+1))

# pdf_ff['weekend_or_not'] = pdf_ff.index.map(lambda x: True if x.weekday() > 4 else False)
# pdf_ff.loc[80, 'call_log'] = 3177
# pdf_ff.loc[142, 'call_log'] = 3744
pdf_ff.loc[pd.Timestamp('2017-07-04'),'call_log'] = 3177
pdf_ff.loc[pd.Timestamp('2017-09-04'),'call_log'] = 3744


for i in range(0, 10):
	col1 = 'application_delay_'+ str(i+1) + '_weekday'
	pdf_ff[col1] = pdf_ff.application.shift(i+1)

for i in range(4, 10):
	col1 = 'call_log_delay_'+ str(i+1) + '_weekday'
	pdf_ff[col1] = pdf_ff.call_log.shift(i+1)

for i in range(4):
	col = 'call_log_prior_'+ str(i+1) + '_week'
	pdf_ff[col] = pdf_ff.call_log.shift(5*(i+1))

pdf_ff['avg_4_weeks'] = pdf_ff[['call_log_prior_1_week', 'call_log_prior_2_week', 'call_log_prior_3_week', 'call_log_prior_4_week']].mean(1)
dict_weekday = {0: 3186.75, 1: 3101.83871, 2: 2877.548387, 3: 2863.387097, 4: 2681.193548}

pdf_ff['average'] = pdf_ff.weekday.map(lambda x: dict_weekday[x])
pdf_ff.to_csv("Desktop/call_log_and_application analysis/non arrangements/pdf_ff.csv")

# test = pd.read_csv("Desktop/call_log_and_application analysis/predicted_raw_data.csv")
test = pdf_ff.loc[datetime(2017,11,4):datetime(2017,11,20)]
# INDICATOR_COLS = test.columns
test1 = pdf_ff.loc[datetime(2017,11,16):datetime(2017,11,20)]

for i in range(len(test)):
    print(predict(test.iloc[i,:]))

for i in range(len(test1)):
    print(predict(test1.iloc[i,:]))



test.head()



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
    round_call_log_delay_5_weekday = np.float32(row[u'call_log_delay_5_weekday'])
    round_call_log_delay_8_weekday = np.float32(row[u'call_log_delay_8_weekday'])
    round_weekday = np.float32(row[u'weekday'])
    round_call_log_prior_4_week = np.float32(row[u'call_log_prior_4_week'])
    round_call_log_delay_7_weekday = np.float32(row[u'call_log_delay_7_weekday'])
    round_application_delay_2_weekday = np.float32(row[u'application_delay_2_weekday'])
    round_application_delay_1_week = np.float32(row[u'application_delay_1_week'])
    round_avg_4_weeks = np.float32(row[u'avg_4_weeks'])
    round_average = np.float32(row[u'average'])
    round_application_delay_10_weekday = np.float32(row[u'application_delay_10_weekday'])
    round_application_delay_9_weekday = np.float32(row[u'application_delay_9_weekday'])
    round_application_delay_3_week = np.float32(row[u'application_delay_3_week'])
    round_call_log_delay_6_weekday = np.float32(row[u'call_log_delay_6_weekday'])
    round_call_log_delay_9_weekday = np.float32(row[u'call_log_delay_9_weekday'])
    round_application_delay_3_weekday = np.float32(row[u'application_delay_3_weekday'])
    round_application_week = np.float32(row[u'application_week'])
    round_application_delay_4_weekday = np.float32(row[u'application_delay_4_weekday'])
    round_application_delay_8_weekday = np.float32(row[u'application_delay_8_weekday'])
    round_application_delay_2_week = np.float32(row[u'application_delay_2_week'])
    round_call_log_prior_3_week = np.float32(row[u'call_log_prior_3_week'])
    round_application_delay_7_weekday = np.float32(row[u'application_delay_7_weekday'])
    round_call_log_delay_10_weekday = np.float32(row[u'call_log_delay_10_weekday'])
    return sum([
        3108.6699950,
            -9.3861486254047949984 * (round_weekday),
         -0.0041394002837281708429 * (round_application_delay_7_weekday),
          0.0025260860554568906854 * (round_call_log_delay_9_weekday),
         0.00095518574876786552404 * (round_call_log_delay_10_weekday),
         0.00016382871659131130449 * (round_call_log_prior_3_week),
             3.5208813162448748812 * (round_call_log_delay_9_weekday > 2684.5 and 
                                     round_call_log_delay_10_weekday > 3359.5 and 
                                     round_avg_4_weeks > 2729.5),
             66.715347869937957626 * (round_call_log_delay_9_weekday > 2913.0 and 
                                     round_call_log_delay_10_weekday > 3069.5 and 
                                     round_call_log_prior_3_week > 2651.5 and 
                                     round_average > 2772.29052734375),
             45.397364604904716145 * (round_application_week > 21441.5 and 
                                     round_call_log_delay_9_weekday > 2684.5 and 
                                     round_call_log_prior_3_week > 2491.5 and 
                                     round_avg_4_weeks > 2937.625),
             23.079199548671951447 * (round_weekday <= 3.5 and 
                                     round_application_week > 20276.5 and 
                                     round_call_log_prior_3_week > 2651.5 and 
                                     round_call_log_prior_4_week <= 3391.5),
             93.813673816908874414 * (round_weekday <= 3.5 and 
                                     round_application_week > 21070.5 and 
                                     round_call_log_delay_9_weekday > 2684.5 and 
                                     round_call_log_prior_3_week > 2651.5),
            -61.858304252604114026 * (round_call_log_prior_3_week <= 2481.5),
             46.242153722238803937 * (round_weekday <= 3.5 and 
                                     round_application_week > 20276.5 and 
                                     round_avg_4_weeks > 3099.75),
             27.403204164083426519 * (round_call_log_delay_7_weekday > 2508.5 and 
                                     round_call_log_delay_9_weekday > 2684.5 and 
                                     round_call_log_prior_3_week > 2770.5 and 
                                     round_average > 2772.29052734375),
             11.177743110882067512 * (round_application_week > 20276.5 and 
                                     round_application_delay_10_weekday > 3871.5 and 
                                     round_call_log_prior_3_week > 2651.5 and 
                                     round_avg_4_weeks > 3096.125),
            -49.067224942525427878 * (round_call_log_delay_6_weekday <= 2759.0 and 
                                     round_call_log_prior_4_week <= 3494.0),
            -26.808470750948288241 * (round_weekday > 3.5 and 
                                     round_application_delay_1_week <= 23095.0 and 
                                     round_application_delay_9_weekday <= 4322.5),
           -0.47197560371671443136 * (round_application_delay_9_weekday <= 4322.5 and 
                                     round_average <= 2772.29052734375),
             35.330373001532656474 * (round_weekday <= 3.5 and 
                                     round_call_log_delay_6_weekday > 2373.5 and 
                                     round_call_log_delay_10_weekday > 2949.0 and 
                                     round_call_log_prior_3_week > 2821.5),
             15.967417358171632102 * (round_call_log_delay_7_weekday > 2491.5 and 
                                     round_call_log_delay_8_weekday > 2553.0 and 
                                     round_call_log_prior_4_week > 3125.5 and 
                                     round_average > 2772.29052734375),
             -40.01081013409574183 * (round_application_delay_1_week <= 23095.0 and 
                                     round_application_delay_2_weekday > 3627.5 and 
                                     round_call_log_delay_8_weekday <= 3494.0 and 
                                     round_call_log_prior_3_week <= 3101.0),
           -0.19325212707702701942 * (round_weekday > 3.5 and 
                                     round_application_delay_8_weekday <= 5075.0),
            -32.683816018363202716 * (round_application_delay_3_weekday > 4272.0 and 
                                     round_call_log_delay_6_weekday <= 2802.5 and 
                                     round_call_log_delay_9_weekday <= 2882.5),
             7.5547736866675077039 * (round_application_delay_10_weekday <= 5422.0 and 
                                     round_call_log_delay_5_weekday <= 3565.0 and 
                                     round_call_log_delay_9_weekday > 2440.5 and 
                                     round_call_log_prior_4_week > 3132.5),
            -3.5622674903235083121 * (round_application_week > 20276.5 and 
                                     round_application_delay_2_weekday > 3627.5 and 
                                     round_application_delay_8_weekday <= 4990.5 and 
                                     round_call_log_delay_10_weekday <= 3377.0),
            -84.199802926417191884 * (round_application_delay_2_weekday > 3627.5 and 
                                     round_call_log_delay_8_weekday <= 3178.0 and 
                                     round_call_log_prior_3_week <= 3620.0 and 
                                     round_call_log_prior_4_week <= 3494.0),
            0.76123589645174694063 * (round_application_week > 20764.5 and 
                                     round_application_delay_3_weekday > 4158.5 and 
                                     round_application_delay_10_weekday <= 5274.0 and 
                                     round_avg_4_weeks > 2893.5),
            -70.583343448131032005 * (round_weekday > 0.5 and 
                                     round_application_delay_3_weekday > 3998.5 and 
                                     round_application_delay_10_weekday <= 5274.0 and 
                                     round_call_log_delay_8_weekday <= 3494.0),
            -10.978218347038875891 * (round_application_delay_2_weekday > 3627.5 and 
                                     round_application_delay_7_weekday > 3767.5 and 
                                     round_application_delay_10_weekday <= 5422.0 and 
                                     round_average <= 2772.29052734375),
            -5.7024648456861566359 * (round_application_delay_2_weekday > 3574.0 and 
                                     round_application_delay_3_weekday <= 5657.0 and 
                                     round_application_delay_8_weekday <= 4655.0 and 
                                     round_call_log_delay_10_weekday <= 3377.0),
             99.792910009818911021 * (round_application_delay_7_weekday > 3767.5 and 
                                     round_application_delay_8_weekday > 3709.0 and 
                                     round_application_delay_10_weekday > 3708.5 and 
                                     round_call_log_delay_6_weekday > 2802.5),
            -26.715831735180692874 * (round_application_delay_1_week <= 23095.0 and 
                                     round_application_delay_4_weekday <= 4968.0 and 
                                     round_application_delay_10_weekday <= 4096.5 and 
                                     round_call_log_prior_3_week <= 3359.5),
            0.77438475968163644758 * (round_application_week > 20276.5 and 
                                     round_call_log_delay_10_weekday > 3377.0),
             30.813423124377340656 * (round_application_week > 20276.5 and 
                                     round_application_delay_3_week <= 22471.5 and 
                                     round_application_delay_10_weekday > 4140.5 and 
                                     round_call_log_prior_3_week <= 3359.5),
             52.433317819742939037 * (round_application_delay_2_week <= 23095.0 and 
                                     round_application_delay_4_weekday <= 5054.5 and 
                                     round_call_log_delay_5_weekday <= 3246.5 and 
                                     round_call_log_delay_6_weekday > 2373.5),
            0.11051403180736010801 * (round_weekday <= 2.5 and 
                                     round_call_log_delay_5_weekday <= 3869.5 and 
                                     round_call_log_delay_7_weekday <= 2929.5 and 
                                     round_call_log_delay_8_weekday <= 3178.0),
            -27.839264023567857009 * (round_application_delay_2_weekday > 3507.0 and 
                                     round_application_delay_7_weekday > 4001.5 and 
                                     round_application_delay_8_weekday <= 4849.5),
             7.1481733114530863205 * (round_application_delay_3_weekday <= 5657.0 and 
                                     round_application_delay_4_weekday <= 4493.5 and 
                                     round_call_log_delay_10_weekday > 2955.5)    ])
