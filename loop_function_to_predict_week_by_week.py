def date_digitize(x):
    for i in range(len(name_c_com.mail_date)-1):
        if (x > name_c_com.mail_date[i]) & (x <= name_c_com.mail_date[i+1]):
            return i
    if x > name_c_com.mail_date[len(name_c_com.mail_date)-1]:
        return len(name_c_com.mail_date) - 1

name_c_com = dff.copy()

df = day_t.copy()

def outgrow_by_7(df):
    for i in range(7):
        df = df.append({'time_by_day': df.iloc[-1]['time_by_day']+timedelta(days=1), 'call_nums': 0}, ignore_index=True)
    day_t = df.copy()
    day_t['mail_volume_1_most_recent_prior'] = day_t.time_by_day.map(lambda x: date_digitize(x))
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
    time_gap_point = day_t.time_by_day.iloc[-1] - timedelta(days=7)
    data_waited_to_predict = day_t[day_t.time_by_day > time_gap_point]
    data_waited_to_predict['prediction'] = data_waited_to_predict.apply(lambda x: predict(x), axis = 1)
    new_day_t = data_waited_to_predict[['time_by_day', 'prediction']]
    new_day_t_head = day_t[['time_by_day', 'call_nums']]
    df_head = new_day_t_head[new_day_t_head.time_by_day <= time_gap_point]
    df_tail = new_day_t[new_day_t.time_by_day > time_gap_point]
    df_tail.columns = df_head.columns
    df_tmp = pd.concat([df_head, df_tail], axis = 0)
    return df_tmp

