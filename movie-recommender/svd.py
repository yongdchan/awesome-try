#coding:utf-8
################################################################################
#                  SVD based Movie Recommender System:                         #
################################################################################

# 加载python模块
import csv
import math
import os
import sys
import time
import numpy as np
from numpy import *

# 余弦相似度计算
def cosine_similarity(v1, v2):
  num = float(np.dot(v1, v2))
  denom = np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2))
  if denom == 0.0:
    return 0.0
  else:
    return 0.5+0.5*(num/denom)

# 计算皮尔逊相性
def pear_similarity(v1, v2):
  n = len(v1)
  if len(v1) < 3: return 1.0
  sum1 = sum([item for item in v1])
  sum2 = sum([item for item in v2])
  sum1_sq = sum([item*item for item in v1])
  sum2_sq = sum([item*item for item in v2])
  psum = np.dot(v1, v2)
  num = psum - (sum1*sum2)/n
  dem = np.sqrt((sum1_sq - math.pow(sum1, 2)/n) * (sum2_sq - math.pow(sum2, 2)/n))
  if dem == 0: return 0.0
  return 0.5+0.5*(num/dem)

# moveRecommender 类声明
class MovieRecommender:
  __n_rows = None
  __n_cols = None
  __um_mat = None
  __us_mat = None
  __svt_mat = None
  item_sim_dict = None

  # 类初始化函数
  def __init__(self, n_rows, n_cols, um_mat):

    self.__n_rows =  n_rows
    self.__n_cols = n_cols
    self.__um_mat = um_mat

  # 对缺失值的补充处理
  def normalize_uv_matrix(self):
    normalize_mat = self.__um_mat.copy()
    # 用列均值填充未打分的缺失值
    c = 0
    for line in normalize_mat.T:
      cav = float(np.sum(line)) / float(self.__n_rows)
      for r in range(0, len(line)):
        if line[r] == 0.0:
          normalize_mat[r, c] = cav
      c += 1

    # 每个元素减去其行均值
    r = 0
    for line in normalize_mat:
      rav = float(np.sum(line)) / float(self.__n_cols)

      for c in range(0, len(line)):
        normalize_mat[r, c] -= rav
      r += 1
    return normalize_mat

  # 评分矩阵svd分解, k为分类因子
  def svd_decomposition(self, data_mat, k):
    u_mat, s_arr, vt_mat = np.linalg.svd(data_mat)
    u_mat = np.delete(u_mat, s_[k:], axis=1)
    vt_mat = np.delete(vt_mat, s_[k:], axis=0)
    s_arr = s_arr[:k]
    s_mat = np.diag(s_arr)
    s_mat = np.sqrt(s_mat)
    self.__us_mat = np.dot(u_mat, s_mat)
    self.__svt_mat = (np.dot(s_mat, vt_mat)).T

  # 计算物品之间的相似性矩阵
  def get_items_similarity(self, sim_meas):
    item_sim_dict = {}
    for i in range(0, self.__n_cols):
      sim_dict  = {}
      r = 0
      for row in self.__svt_mat:
        if i != r:
          cs = sim_meas(self.__svt_mat[i], row)
          sim_dict[r] = cs
        r += 1
      item_sim_dict[i] = sim_dict
    self.item_sim_dict = item_sim_dict

  def svd_recommend_train(self, k, sim_meas):
    # 数据预处理
    normalize_mat = self.normalize_uv_matrix()
    # svd 分解
    self.svd_decomposition(normalize_mat, k)
    # 求item相似矩阵
    self.get_items_similarity(sim_meas)

  # compute top n movies which can be recommended to user-id
  def __compute_top_n_movies(self, u_dict, user_id):

    movie_arr = np.zeros(self.__n_cols)
    c_dict = u_dict[self.user_id-1]   # cosine similarity of user i is recor-
                                        # -ded in loc i-1
    for k, v in c_dict:
       movie_arr = np.add(movie_arr, self.__um_mat[k])

    m_dict = {}
    for i in range(0,self.__n_cols):
      m_dict[i] = movie_arr[i]

    m_dict = sorted(m_dict.items(), key = lambda arg: arg[1], reverse = True)
    m_dict = m_dict[0:self.__top_n]

    top_n_list = []
    for k, v in m_dict:
       top_n_list.append(k+1) # we need to add 1, because 0th col corresponds
                              # movie 1, etc 
    return top_n_list

  # compute k nearest neighbors of each user                                   #
  def k_nearest_neighbor_users(self):

    u_dict = {}
    for i in range(0, self.__n_rows):
      r = 0
      c_dict = {}
      for row in self.__us_mat:
        if i != r:
          cs = cosine_similarity(self.__us_mat[i], row)
          c_dict[r] = cs
        r += 1
      c_dict = sorted(c_dict.items(), key = lambda arg: arg[1], reverse = True)
      c_dict = c_dict[0:self.knn]
      u_dict[i] = c_dict

    return u_dict


  # 评分预测函数
  # 预测指定用户user_id对电影movie_id 的评分
  def rating_guess(self, user_id, movie_id):
    sim_total = 0.0
    pre_rating = 0.0
    for j in range(self.__n_cols):
      user_rating = self.__um_mat[user_id, j]
      #user_rating = ori_mat[user_id, j]
      if user_rating == 0 or j == movie_id: continue
      similarity = self.item_sim_dict[movie_id][j]
      sim_total += similarity
      pre_rating += similarity * user_rating
    if sim_total == 0:
      p_score = 0
    else:
      p_score = pre_rating / sim_total
    if p_score < 1:
      p_score = 1
    elif p_score > 5:
      p_score = 5
    return p_score

  # Top-N 电影推荐函数
  def top_n_recommend(self, user_id, top_n):
    movie_arr = self.__um_mat[user_id]
    unrated_items = [id for id in range(self.__n_cols) if movie_arr[id] == 0]
    if len(unrated_items) == 0:
      return 'you have enjoy everything...'
    item_score = []
    for item in unrated_items:
      pre_score = self.rating_guess(user_id, item)
      item_score.append((item, pre_score))
    item_score = sorted(item_score, key=lambda arg: arg[1], reverse = True)

    top_n_list = item_score[:top_n]

    return top_n_list

# 打印top-N 推荐
def print_recommended_movies(user_id, recommended_list, movie_data_base):
  if len(recommended_list) == 0:
    print "No movies to recommend for user_id:", user_id
  else:
    print "Top-N\t\tMovie name\t\tRelease date"
    print "************************************************************************"
    i = 1
    for k, v in recommended_list:
      item = movie_data_base[k]
      print i," ",str(item[1]).ljust(50),item[2]#,round(v,1)
      i += 1
    print "************************************************************************"

# 参数获取
def get_few_essential_parameters():

  k = 19   # size of s matrix of svd decomposition is to be set to k x k 
  knn = 14 #  k nearest neighbor users
  n = 10   #  top n movies to be recommended to user-id
  return k, knn, n

# 构造用户-电影 二维矩阵
def construct_user_movie_matrix(user_data_base, movie_data_base, \
                                rating_data_base):
   n_rows = int(len(user_data_base))
   n_cols = int(len(movie_data_base))
   um_matrix = np.zeros((n_rows, n_cols))
   # 在user-movie矩阵中插入rating data 
   for line in rating_data_base:
     u = int(line[0])
     m = int(line[1])
     r = int(line[2])
     if r != 0:
       um_matrix[u-1, m-1] = r
   return n_rows, n_cols, um_matrix

# 获取需要推荐的用户ID
def get_user_id(user_count):
  print "输入需要推荐的用户ID[1," + str(user_count) + "] 输入其他任何数字退出系统"
  user_id = int(input())
  if user_id < 1 or user_id > user_count:
    print "Bye~~, exiting system, enjoy youself"
    sys.exit()
  return user_id

# 训练数据读取
def read_data_base_files(user_file, movie_file, rating_file):
  # 读取user data
  try:
    user_data_base = list(csv.reader(user_file, delimiter='|'))
  except csv.Error:
    print("\nError: in reading user data base file.\n")
    sys.exit()

  # 读取movie data
  try:
    movie_data_base = list(csv.reader(movie_file, delimiter='|'))
  except csv.Error:
    print("\nError: in reading movie data base file.\n")
    sys.exit()

  # 读取rating data
  try:
    rating_data_base = list(csv.reader(rating_file, delimiter='\t'))
  except csv.Error:
    print("\nError: in reading rating data base file.\n")
    sys.exit()

  return user_data_base, movie_data_base, rating_data_base

# 训练数据文件打开
def open_data_base_files():
  try:
    user_file = open("./data/u.user", 'r')
  except IOError:
    print "Error: The user data base file 'u.user' does not exist, exiting "
    sys.exit()
  try:
    movie_file = open("./data/u.item", 'r')
  except IOError:
    print "Error: The movie data base file 'u.item' does not exist, exiting "
    sys.exit()
  try:
    rating_file = open("./data/u.base", 'r')
  except IOError:
    print "Error: The rating data base file 'u.data' does not exist, exiting "
    sys.exit()

  return user_file, movie_file, rating_file

# 关闭文件
def close_data_base_files(user_file, movie_file, rating_file):
  user_file.close()
  movie_file.close()
  rating_file.close()

# 电影TOPN推荐函数
def svd__recommender(sim_meas):
  user_file, movie_file, rating_file = open_data_base_files()
  user_data_base, movie_data_base, rating_data_base = read_data_base_files( \
                                             user_file, movie_file, rating_file)
  n_rows, n_cols, um_matrix = construct_user_movie_matrix(user_data_base, \
                                              movie_data_base, rating_data_base)

  # 构建MovieRecommender对象
  movie_recommender = MovieRecommender(n_rows, n_cols, um_matrix)
  # svd based recommend system 训练
  movie_recommender.svd_recommend_train(14, sim_meas)

  # 输入用户ID, 获取top-N推荐
  while True:
    user_id = int(get_user_id(len(user_data_base)))
    print "请输入您需要的TOP-N值："
    top_n = int(input())
    recommended_list = movie_recommender.top_n_recommend(user_id-1, top_n)
    # 打印Top-N 推荐
    print_recommended_movies(user_id, recommended_list, movie_data_base)

  close_data_base_files(user_file, movie_file, rating_file)

# 算法验证函数
def validation(sim_meas):
  start_time = time.clock()

  user_file, movie_file, rating_file = open_data_base_files()
  user_data_base, movie_data_base, rating_data_base = read_data_base_files( \
                                             user_file, movie_file, rating_file)
  n_rows, n_cols, um_matrix = construct_user_movie_matrix(user_data_base, \
                                              movie_data_base, rating_data_base)

  # 构建MovieRecommender对象
  movie_recommender = MovieRecommender(n_rows, n_cols, um_matrix)
  # svd based recommend system 训练
  movie_recommender.svd_recommend_train(14, sim_meas)
  close_data_base_files(user_file, movie_file, rating_file)
  end1 = time.clock()
  print "Total training time is %f" % (end1 - start_time)
  try:
    test_file = open("./data/u.test", 'r')
  except IOError:
    print("\nError: The test data 'u.test' does not exist, exiting...\n")
    sys.exit()
  
  try:
    test_data_base = list(csv.reader(test_file, delimiter='\t'))
  except csv.Error:
    print("\nError: in reading test data base file, exiting..\n")
    sys.exit()

  cnt = 0
  rmse = 0.0
  for entries in test_data_base:
      cnt += 1
      predict_score = movie_recommender.rating_guess(int(entries[0])-1, \
                                                    int(entries[1])-1)
      guess_diff = predict_score - int(entries[2])
      rmse += guess_diff * guess_diff
  test_file.close()
  rmse_result = math.sqrt(rmse/cnt)
  end2 = time.clock()
  print "Total validation time is %f" % (end2 - start_time)
  print "The RMSE of this algorithm is %f" % rmse_result

# 主入口函数
if __name__ == "__main__":
  #validation(cosine_similarity)
  svd__recommender(cosine_similarity)
