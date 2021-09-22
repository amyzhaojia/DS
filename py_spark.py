import matplotlib
from pyspark import SparkContext
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd

# 测试pyspark
sc = SparkContext('local[3]','First_Spark_App')
# data = sc.textFile()
t1 = time.time()
data = sc.textFile('./co_05_diff_1min.csv') #.map(lambda line:line.split(",")).map(lambda record:(record[0],record[1],record[2])) 
t2 = time.time()
# data = pd.read_csv('./co_05_diff_1min.csv')
t3 = time.time()
print("t_spark:%f,t_pandas:%f" %(t2-t1,t3-t2))
numPurchases = data.count()
uniqueUsers = data.map(lambda record:record[0]).distinct().count()
totalRevenue = data.map(lambda record:float(record[2])).sum()
products = data.map(lambda record: (record[1],1.0)).reduceByKey(lambda a,b:a+b).collect()
mostpopular = sorted(products,key=lambda x:x[1],reverse=True)[0]
print('Total purcharses:%d' %numPurchases)

# # spark 数据统计以及展示
user_data = sc.textFile(r'E:\Data\ml-100k\ml-100k\u.user')
# # print(user_data.first())
user_fields = user_data.map(lambda line:line.split("|"))
num_users = user_fields.map(lambda fields:fields[0]).count()
# num_genders = user_fields.map(lambda fields:fields[2]).distinct().count()
# num_occupations = user_fields.map(lambda fields:fields[3]).distinct().count()
# num_zipcodes = user_fields.map(lambda fields:fields[4]).distinct().count()
# print("Users:%d, genders:%d, occupations:%d, ZIP codes:%d" %(num_users,num_genders,num_occupations,num_zipcodes))
# ages = user_fields.map(lambda fields:int(fields[1])).collect()
# # plt.hist(ages,bins=20,color='lightblue')
# # fig = plt.gcf()
# # fig.set_size_inches(16,10)
# count_by_occupation = user_fields.map(lambda fields:(fields[3],1)).reduceByKey(lambda x,y:x+y).collect()
# x_axis1 = np.array([c[0] for c in count_by_occupation])
# y_axis1 = np.array([c[1] for c in count_by_occupation])
# x_axis = x_axis1[np.argsort(y_axis1)]
# y_axis = y_axis1[np.argsort(y_axis1)]  # 按y轴值排序
# pos = np.arange(len(x_axis))
# width = 1.0
# ax = plt.axes()
# ax.set_xticks(pos+(width/2))
# ax.set_xticklabels(x_axis)
# plt.bar(pos,y_axis,width,color='lightblue')
# plt.xticks(rotation=30)
# fig = plt.gcf()
# fig.set_size_inches(16,10)
# count_by_occupation2 = user_fields.map(lambda fields:fields[3]).countByValue()
# print(dict(count_by_occupation))
# print(dict(count_by_occupation2))
# plt.show()

# # saprk数据异常处理、特征提取
movie_data = sc.textFile(r'E:\Data\ml-100k\ml-100k\u.item')
num_movies = movie_data.count()
# def convert_year(x):
#     try:
#         return int(x[-4:])
#     except:
#         return 1900
# movie_fields = movie_data.map(lambda lines:lines.split("|"))
# years = movie_fields.map(lambda fields:fields[2]).map(lambda x:convert_year(x))
# years_filtered = years.filter(lambda x:x != 1900)
# movie_ages = years_filtered.map(lambda yr: 1998-yr).countByValue()
# values = movie_ages.values()
# bins = movie_ages.keys()
# plt.hist(values,bins=np.sort(np.array(list(bins))),color='lightblue')  
# fig = plt.gcf()
# fig.set_size_inches(16,10)
# plt.show()
rating_data_raw = sc.textFile(r'E:\Data\ml-100k\ml-100k\u.data')
num_ratings = rating_data_raw.count()
rating_data = rating_data_raw.map(lambda line:line.split("\t"))
ratings = rating_data.map(lambda fields:int(fields[2]))
max_rating = ratings.reduce(lambda x,y:max(x,y))
min_rating = ratings.reduce(lambda x,y:min(x,y))
mean_rating = ratings.reduce(lambda x,y:x+y)/num_ratings
median_rating = np.median(ratings.collect())
ratings_per_user = num_ratings/num_users
ratings_per_movie = num_ratings/num_movies
# print(min_rating,max_rating,mean_rating,median_rating,ratings_per_user,ratings_per_movie)
# print(ratings.stats())
count_by_rating = ratings.countByValue()



# from operator import add
# import random
# from pyspark import SparkConf, SparkContext
# sc = SparkContext('local')
# NUM_SAMPLES = 100000
# def inside(p):
#     x, y = random.random(), random.random()
#     return x*x + y*y < 1
# count = sc.parallelize(range(0, NUM_SAMPLES)).filter(inside).count()
# print ("Pi is roughly %f" % (4.0 * count / NUM_SAMPLES))

# from pyspark.sql import SparkSession
# spark = SparkSession.builder\
#     .appName('My_App')\
#     .master('local')\
#     .getOrCreate()
# df = spark.read.csv('./Staff_forecast.csv',header=True)
# df.printSchema()