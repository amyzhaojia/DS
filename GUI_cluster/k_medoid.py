import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
 
def im_txt(file):
    """
    读取数据
    """
    data=np.loadtxt(file,dtype=np.float32)
    return data
 
def initianlize_centers(n_clusters):
    """初始化，生成随机聚类中心"""
    n_data=lendata()
    centers=[]  #聚类中心位置信息例：[101,205,5,3,7]
    i=0
    while i<n_clusters:
        temp=random.randint(0,n_data-1)
        if temp not in centers:
            centers.append(temp)
            i=i+1
        else:
            pass
    return centers
 
def clus_process(centers,data):
    """根据聚类中心进行聚类"""
    result_clusters=[]
    centers=np.array(centers)
    """遍历每个样本"""
    for i in range(0,len(data)):
        uni_temp=[] #临时存储距离数据
        for j in centers:
            temp=np.sqrt(np.sum(np.square(data[i]-data[j])))
            uni_temp.append(temp)
        c_min=min(uni_temp) #距离最小值
        result_clusters.append(uni_temp.index(c_min))  #距离最小值所在位置即为归属簇
 
    return result_clusters
 
def chose_centers(result_clusters,n_clusters):
    global c_temp
    centers=[]
    for i in range(0,n_clusters):  #逐个簇进行随机
        temp=[]  #记录每个簇样本在data中的位置
        for j in range(0,len(result_clusters)):   #遍历每个样本
            if result_clusters[j]==i:     #寻找簇i的样本
                temp.append(j)
        try:
            c_temp=random.sample(temp,1)   #在样本中随机取一个值作为新的聚类中心
        except:
            print("sample bug")
            print(temp)
        centers.append(c_temp[0])
 
    return centers
 
def count_E(centers_new,data,result_clusters_new):
    """计算价值函数"""
    E=0
    for i in range(0,len(centers_new)):
        for j in range(0,len(data)):
            if result_clusters_new[j]==i:
                temp=np.sqrt(np.sum(np.square(data[j]-data[centers_new[i]])))
                E+=temp
    return E
 
def KMedoids(n_clusters,data,max_iter):
    """初始化"""
    centers=initianlize_centers(n_clusters)
    """根据随机中心进行聚类"""
    result_clusters=clus_process(centers,data)
    """重新选择聚类中心,并比较"""
    xie=0  #计数器
    E=5*5000
    """
    _old：用来记录上一次的聚类结果
    _new：新一次聚类的结果
    无old和new：输出结果
    """
    while xie<=max_iter:
        centers_new=chose_centers(result_clusters,n_clusters)  #新的聚类中心
        result_clusters_new=clus_process(centers,data)  #新的聚类结果
        """计算价值函数E"""
        E_new=count_E(centers_new,data,result_clusters_new)
        """价值函数变小，则更新聚类中心和聚类结果"""
        if E_new<E:
           centers=centers_new
           result_clusters=result_clusters_new
           E=E_new
           t=""
           y=""
           t=t+"价值函数为:"+str(E)+"\n"
           # print("价值函数为:%s"%E)
           y=y+"聚类中心:"+str(centers)+"\n"
           # print("聚类中心:%s"%centers)
           print(t)
           print(y)
           xie=0
        """阈值计数器"""
        xie=xie+1
 
 
    return centers,result_clusters
 
 
def randomcolor(x):
    """随机生成十六进制编码"""
    colors=[]
    i=0
 
    while i<x:
        colorArr = ['1','7','A','F']
#        colorArr = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
        color = ""
        j=0
        while j<6:
            color += colorArr[random.randint(0,3)]
            j=j+1
        color="#"+color
        if color in colors:
            continue
        else:
            colors.append(color)
            i=i+1
    return colors
 
def lendata():
    file="a_data_set.txt"
    data=im_txt(file)
    n_data=len(data)
    return n_data