import pandas as pd
import datetime
import numpy as np
from pandas import DataFrame
import csv

def create_assist_date(datestart = None,dateend = None):
	# 创建日期辅助表

	if datestart is None:
		datestart = '2016-01-01'
	if dateend is None:
		dateend = datetime.datetime.now().strftime('%Y-%m-%d')

	# 转为日期格式
	datestart=datetime.datetime.strptime(datestart,'%Y-%m-%d')
	dateend=datetime.datetime.strptime(dateend,'%Y-%m-%d')
	date_list = []
	date_list.append(datestart.strftime('%Y-%m-%d'))
	while datestart<dateend:
		# 日期叠加一天
	    datestart+=datetime.timedelta(days=+1)
		# 日期转字符串存入列表
	    date_list.append(datestart.strftime('%Y-%m-%d'))
	return date_list


df = pd.read_csv('../MyPrediction/Classification/data/india_beforeOmicron.csv', on_bad_lines='skip')
# df_second = pd.read_csv('../Classification/data/california_second.csv')
# df_third = pd.read_csv('../Classification/data/california_third.csv')
df_confirmed = pd.read_csv('../MyPrediction/RegressionModel/data/india_part.csv')
# total_cases = df_confirmed['total_cases'][640:].to_list()
# new_cases_smoothed = df_confirmed['new_cases_smoothed'][640:].to_list()
# total_deaths = df_confirmed['total_deaths'][640:].to_list()
# new_deaths_smoothed = df_confirmed['new_deaths_smoothed'][640:].to_list()
# stringency_index = df_confirmed['stringency_index'][640:].to_list()

polar_list = df['polar'].to_list() # 每天500条推文的情绪极性
# polar_list_second = df_second['polar'].to_list()
# polar_list_third = df_third['polar'].to_list()
date_list = create_assist_date('2020-03-01', '2021-11-30')
# date_list_second = create_assist_date('2021-03-01', '2021-11-30')
# date_list_third = create_assist_date('2021-12-01', '2022-08-31')


neg_count,neu_count,pos_count = 0,0,0

neg_list = []
neu_list = []
pos_list = []

for i in range(0,640):
	neg_count = 0.00
	neu_count = 0.00
	pos_count = 0.00
	for j in range(i*50,i*50+50):
		if polar_list[j] == 'roberta_neg' :
			neg_count +=1
		if polar_list[j] == 'roberta_neu' :
			neu_count += 1
		if polar_list[j] == 'roberta_pos' :
			pos_count += 1
	neg_list.append(neg_count/50.0)
	neu_list.append(neu_count/50.0)
	pos_list.append(pos_count/50.0)



# print(polar_dict_list)
df_los = pd.DataFrame({'date':date_list,'neg':neg_list,'neu':neu_list,'pos':pos_list})
df_los.to_csv('../RegressionModel/data/india_beforeOmicron_sentiment.csv',mode='a+')
print('done')
