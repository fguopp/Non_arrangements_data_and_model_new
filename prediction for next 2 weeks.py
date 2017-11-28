predicted_next_week = list()
for i in range(len(test1)):
    predicted_next_week.append(predict(test1.iloc[i,:]))

base = datetime(2017,11,20)
date_list = [base + timedelta(days=x) for x in range(5)]

projection = pd.read_csv("Desktop/call_log_and_application analysis/projections.csv")
projection['timestamp'] = pd.to_datetime(projection.timestamp, format='%m/%d/%y')


proto_df = list()
for i in range(5):
    ssss = projection[(projection.timestamp == date_list[i].date())]['application_num'].iloc[0]
    proto_df.append([date_list[i].date(), ssss, predicted_next_week[i]])

df_1_need_to_append = pd.DataFrame(proto_df, columns=['timestamp', 'application_num', 'call_log_num'])

sdf_1 = df_1.copy()
sdf_1 = sdf_1.drop(sdf_1.index[-1])
tt = df_1_need_to_append.combine_first(sdf_1)

tt = tt.sort_values(['timestamp']).reset_index(drop = True)
tt['weekday'] = tt.timestamp.map(lambda x: x.weekday())

cols = ['timestamp', 'application_num', 'call_log_num', 'weekday']
tt = tt.reindex_axis(cols, axis=1)


t_pseduo = [[datetime(2017,4,12).date(), 3600.5, 3829, 4],\
    [datetime(2017,4,13).date(), 3875.5, 2982, 5], \
    [datetime(2017,4,14).date(), 3306.5, 2449, 6]]

# projection = pd.read_csv("Desktop/call_log_and_application analysis/projections.csv")
# projection['timestamp'] = pd.to_datetime(projection.timestamp, format='%m/%d/%y')
sliced_projection = projection[(projection.timestamp <= datetime.today().date()+timedelta(days=20)) & (projection.timestamp >= datetime(2017, 11, 20))]

for i in range(14):
    apps = sliced_projection[(sliced_projection.timestamp == max(tt.timestamp)+timedelta(days=i+1))]['application_num'].iloc[0]
    t_pseduo.append([max(tt.timestamp)+timedelta(days=i+1), apps, np.nan, np.nan])

t_pseduo = pd.DataFrame(t_pseduo, columns = tt.columns)
p_df = tt.append(t_pseduo).reset_index(drop = True).sort_values(['timestamp'])
p_df.at[188, 'application_num'] = sliced_projection[(sliced_projection.timestamp == datetime(2017,11,20))]['application_num'].iloc[0]

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


test2 = pdf_ff.loc[datetime(2017,11,27):datetime(2017,12,1)]

for i in range(len(test2)):
    print(predict(test2.iloc[i,:]))

