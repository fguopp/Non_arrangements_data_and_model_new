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

# directmail = rawdata("directmail.clr_remail")
# directmail.fetch()
# directmail_copy = directmail.rawdata.copy()
# # directmail_copy['sxn'] = directmail_copy.sxn.astype(str)

# cust = rawdata("all_customer")
# cust.fetch()
# cust_copy = cust.rawdata.copy()

# apps = rawdata("all_allapps")
# apps.fetch()
# apps_copy = apps.rawdata.copy()

del call_log#, directmail, cust, apps

call_log_copy = call_log_copy[call_log_copy.calltype == 'Inbound']

call_log_copy['time_by_hour'] = call_log_copy.timestamp.map(lambda x: str(x.year) + '-' + str(x.month) + '-' + str(x.day) + '-' + str(x.hour))
call_log_copy['time_by_day'] = call_log_copy.timestamp.map(lambda x: str(x.year) + '-' + str(x.month) + '-' + str(x.day))
call_log_copy['hour'] = call_log_copy.timestamp.map(lambda x: x.hour)
call_log_copy['weekday'] = call_log_copy.timestamp.map(lambda x: x.dayofweek)


hour_t = call_log_copy.groupby('time_by_hour').size()
day_t = call_log_copy.groupby('time_by_day').size()
# day_t = call_log_copy.groupby('time_by_day').size()
t = call_log_copy.groupby(['hour', 'time_by_day']).size().reset_index()
t.columns = ['hour', 'time_by_day', 'nums']

tt = t[t.hour == 0]
for h in t.hour.unique():
    if h != 0:
        temp = t[t.hour == h]
        tt = pd.concat([tt, temp], axis = 0)

        tt = pd.concat(tt, temp, on = ['time_by_day'], how = 'outer', axis = 1)

st = pd.pivot_table(t, values = 'nums', index = ['time_by_day'], columns = ['hour'], aggfunc = np.sum)
st = st.reset_index()
st['total'] = st.sum(axis = 1)
st.corr()
st['call_date'] = st['time_by_day'].map(lambda x: datetime.strptime(x, "%Y-%m-%d").date())

tt.to_csv("Desktop/direct mail analysis/tt.csv")



test = pd.read_csv("Desktop/direct mail analysis/untitled/test.csv")
del test["Unnamed: 0"]
test['call_date'] = test['call_date'].map(lambda x: datetime.strptime(x, "%Y-%m-%d").date())


merged_data = pd.merge(st, test, on = 'call_date')
merged_data = merged_data.fillna(0)


import numpy as np
import pandas as pd
import time
import re
from datetime import datetime
import sys
import os
import re
import calendar


test = pd.read_csv("Desktop/direct mail analysis/new/new_call_volumn_inputs.csv")
# INDICATOR_COLS = test.columns

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import datetime as dt

df_ohlc = test[['call_nums', 'weekday', 'days_from_last_1_maildate']]
# ax = df_ohlc.boxplot(column='call_nums', by='weekday', grid=False)
ax = df_ohlc.boxplot(column='call_nums', by='days_from_last_1_maildate', grid=False)


df_ohlc = test[['call_nums', 'weekday', 'days_from_last_3_maildate']]
# ax = df_ohlc.boxplot(column='call_nums', by='weekday', grid=False)
ax = df_ohlc.boxplot(column='call_nums', by='days_from_last_3_maildate', grid=False)
plt.show()

df_ohlc.groupby(['weekday']).describe().to_csv("Desktop/direct mail analysis/new/by_weekday_data_describe.csv")
df_ohlc.groupby(['days_from_last_1_maildate']).describe().to_csv("Desktop/direct mail analysis/new/by_lastmaildate_data_describe.csv")


# from scipy import stats
# b = list()
# for i in range(24):
#     y = merged_data[i].values.reshape((173, 1))
#     x = merged_data[['numbers_y', 'weekday']].values.reshape((173, 2))
#     slope, intercept, r_value, p_value, std_err = stats.mstats.linregress(x,y)
#     b.append([slope, intercept, r_value*r_value, p_value, std_err])

#
#b = list()
#from sklearn import linear_model
#for i in range(24):
#    y = merged_data[i].values.reshape((173, 1))
#    x = merged_data[['numbers_y', 'weekday']].values.reshape((173, 2))
#    regr = linear_model.LinearRegression()
#    # regr = LinearRegression()
#    regr.fit(x, y)
#    b.append([regr.coef_, regr.intercept_, regr.score(x,y)])
#
#bb = list()
#for i in range(len(b)):
#    bb.append([b[i][0][0], b[i][1][0], b[i][2]])
#
#
#combined_regression = pd.DataFrame(bb, columns = ['coefficent', 'intercept', 'r-square'])
#combined_regression.to_csv("Desktop/direct mail analysis/untitled/combined_regression_orig.csv")
#
#
#
#from sklearn import linear_model
#from scipy import stats
#import numpy as np
#
#
#class LinearRegression(linear_model.LinearRegression):
#    """
#    LinearRegression class after sklearn's, but calculate t-statistics
#    and p-values for model coefficients (betas).
#    Additional attributes available after .fit()
#    are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
#    which is (n_features, n_coefs)
#    This class sets the intercept to 0 by default, since usually we include it
#    in X.
#    """
#    def __init__(self, fit_intercept=True, normalize=False, copy_X=True, n_jobs=1):
#        self.fit_intercept = fit_intercept
#        self.normalize = normalize
#        self.copy_X = copy_X
#        self.n_jobs = n_jobs
#    def fit(self, X, y, n_jobs=1):
#        self = super(LinearRegression, self).fit(X, y, n_jobs)
#        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
#        se = np.array([
#            np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X))))
#                                                    for i in range(sse.shape[0])
#                    ])
#        self.t = self.coef_ / se
#        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1]))
#        return self
#
#
#
