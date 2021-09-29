# -*- coding: utf-8 -*-
import sys
import random
import kmeans
import k_medoid as k2d
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import scrolledtext
from PIL import Image,ImageTk
import matplotlib.pyplot as plt
import sklearn
import os
# os.path.join(os.path.dirname(__file__))
 
class GUI(object):
    #布局界面
    def __init__(self):
        #设置初始界面
        self.window=tk.Tk()
        self.window.title('数据聚类系统')
        self.window.geometry('1150x580')
        #导入文件按钮
        self.botton1=tk.Button(self.window, text='加载数据集',bg='green',fg='white',  font=('楷体', 12, 'bold'), width=12, height=1,command=self.openfile)
        self.botton1.place(x=60,y=60)
        #标签配置
        self.label2=tk.Label(self.window, text='簇个数',bg='light blue',fg='white', font=('楷体', 16, 'bold'), width=10, height=1).place(x=10,y=160)
        #导入文件内容的输出显示
        self.label4=tk.Label(self.window, text='导入文件内容如下',font=('楷体', 16, 'bold'), width=16, height=1).place(x=280,y=20)
        #创建结果显示框
        self.text1=scrolledtext.ScrolledText(self.window, height=10, width=30,font=('楷体', 12))
        self.text1.place(x=250,y=60)
        self.text1.bind("<Button-1>",self.clear)
        #各个频繁项集和强关联规则的输出显示
        self.label5=tk.Label(self.window, text='聚类实现',font=('楷体', 16, 'bold'), width=20, height=1).place(x=255,y=290)
        self.label6=tk.Label(self.window, text='聚类可视化',font=('楷体', 16, 'bold'), width=20, height=1).place(x=700,y=20)
        #创建结果显示框
        self.text2=scrolledtext.ScrolledText(self.window, height=10, width=30,font=('楷体', 12))
        self.text2.place(x=250,y=330)
        self.text2.bind("<Button-1>",self.clear)
        #显示导入文件的路径
        self.var0=tk.StringVar()
        self.entry1=tk.Entry(self.window, show=None, width='25', font=('Arial', 10), textvariable=self.var0)
        self.entry1.place(x=10,y=100)
        #自行设置簇个数，个数为2
        self.var1=tk.StringVar()
        self.var1.set('2')
        self.entry2=tk.Entry(self.window, show=None, width='3', font=('Arial', 16), textvariable=self.var1)
        self.entry2.place(x=180,y=160)
        #选择所需算法
        self.btnlist=tk.IntVar()
        self.radiobtn1=tk.Radiobutton(self.window, variable=self.btnlist, value=0, text='K-means聚类算法', font=('bold'),command=self.runkmeans)
        self.radiobtn1.place(x=30,y=240)
        self.radiobtn2=tk.Radiobutton(self.window, variable=self.btnlist, value=1,text='K-中心点聚类算法', font=('bold'), command=self.runkmid)
        self.radiobtn2.place(x=30,y=300)
        self.btnlist.set(0)
        #清空页面按钮
        self.btn2=tk.Button(self.window, bg='green',fg='white', text='清屏', font=('楷体', 12,'bold'), width=6, height=1)
        self.btn2.place(x=80,y=380)
        self.btn2.bind("<Button-1>",self.clear)
        #关闭页面按钮
        self.btn3=tk.Button(self.window, bg='green',fg='white', text='退出', font=('楷体', 12,'bold'), width=6, height=1)
        self.btn3.place(x=80,y=450)
        self.btn3.bind("<Button-1>",self.close)
        self.pilImage = Image.open(r"E:\Data_analysis\GUI_cluster\white.png")
        img=self.pilImage.resize((500,480))
        self.tkImage = ImageTk.PhotoImage(image=img)
        self.label = tk.Label(self.window, image=self.tkImage)
        self.label.place(x=600,y=60)
        #主窗口循环显示
        self.window.mainloop()
 
    #清空所填内容   
    def clear(self,event):
        self.text1.delete("1.0",tk.END)
        self.text2.delete("1.0",tk.END)
        self.pilImage = Image.open(r"E:\Data_analysis\GUI_cluster\white.png")
        img=self.pilImage.resize((500,480))
        self.tkImage = ImageTk.PhotoImage(image=img)
        self.label = tk.Label(self.window, image=self.tkImage)
        self.label.place(x=600,y=60)
        self.label.configure(image = img)
        self.window.update_idletasks()
    #退出系统，对控制台清屏   
    def close(self,event):
        e=tk.messagebox.askokcancel('询问','确定退出系统吗？')
        if e==True:
            exit()
            self.window.destroy() 
    # 恢复sys.stdout     
    def __del__(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
         
    #从输入文本框中获取文本并返回数字列表
    def getCNUM(self):  
          entry_num1 = int(self.var1.get())
          return entry_num1
       
    def openfile(self):
        nameFile = filedialog.askopenfilename(title='打开文件', filetypes=[('txt', '*.txt')])
        self.entry1.insert('insert', nameFile)
         
    def getnamefile(self):
        namefile=self.var0.get()
        return namefile
     
    #加载kmeans所需的数据集      
    def loadDataSet1(self):
        nameFile=self.getnamefile()
        data = np.loadtxt(nameFile,delimiter='\t')
        self.text1.insert("0.0",data)
        return data
     
    #加载k-中点所需的数据集
    def loadDataSet2(self):
        data = []
        for i in range(100):
            data.append(0 + i)
        for i in range(100):
            data.append(1000 + i)
        random.shuffle(data)
        return data
     
    def runkmeans(self):
        dataSet = self.loadDataSet1()
        k = self.getCNUM()
        c=kmeans.randCent(dataSet, k)
        centroids,clusterAssment = kmeans.KMeans(dataSet,k)
        self.text2.insert('insert',c)
        c1,c2,c3,c4=kmeans.showCluster(dataSet,k,centroids,clusterAssment)
        self.text2.insert('insert',c1)
        t0='\n'
        self.text2.insert('insert',t0)
        self.text2.insert('insert',c2)
        self.text2.insert('insert',t0)
        self.text2.insert('insert',c3)
        self.text2.insert('insert',t0)
        self.text2.insert('insert',c4)
        kmeans.showCluster(dataSet,k,centroids,clusterAssment)
        self.pilImage = Image.open("kpic.png")
        img=self.pilImage.resize((500,480))
        self.tkImage = ImageTk.PhotoImage(image=img)
        self.label = tk.Label(self.window, image=self.tkImage)
        self.label.place(x=600,y=60)
        self.label.configure(image = img)
        self.window.update_idletasks()
     
    def runkmid(self):
        data=k2d.im_txt(r"E:/Data_analysis/file/icm_ticket/新建文本文档.txt")
        self.text1.insert("0.0",data)
        data_TSNE = sklearn.manifold.TSNE(learning_rate=100,n_iter=5000).fit_transform(data)
        k=self.getCNUM()
        t='簇中心：\n'
        t1='\n'
        self.text2.insert('insert',t)
        centers,result_clusters = k2d.KMedoids(k,data,10)
        self.text2.insert('insert',centers)
        self.text2.insert('insert',t1)
        color=k2d.randomcolor(k)
        colors = ([color[k] for k in result_clusters])
        color = ['black']
        plt.scatter(data_TSNE[:,0],data_TSNE[:,1],s=10,c=colors)
        plt.title('K-medoids Resul of '.format(str(k)))
        plt.savefig("kpic1.png")
        s1="第一类："
        s2="第二类："
        s3="第三类："
        s4="第四类："
        m=1
        for m in range(len(result_clusters)):
             
            if result_clusters[m]==0:
                s1=s1+str(data[m])+","
            if result_clusters[m]==1:
                s2=s2+str(data[m])+","
            if result_clusters[m]==2:
                s3=s3+str(data[m])+","
            if result_clusters[m]==3:
                s4=s4+str(data[m])+","
        self.text2.insert('insert',s1)
        t1='\n'
        self.text2.insert('insert',t1)
        self.text2.insert('insert',s2)
        self.text2.insert('insert',t1)
        self.text2.insert('insert',s3)
        self.text2.insert('insert',t1)
        self.text2.insert('insert',s4)
        self.pilImage = Image.open("kpic1.png")
        img=self.pilImage.resize((500,480))
        self.tkImage = ImageTk.PhotoImage(image=img)
        self.label = tk.Label(self.window, image=self.tkImage)
        self.label.place(x=600,y=60)
        self.label.configure(image = img)
        self.window.update_idletasks()
            
if __name__ == '__main__':
    GUI()