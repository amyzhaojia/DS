from operator import index
from nltk.corpus.reader.wordnet import Lemma
import pandas as pd
import jieba
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import pos_tag
from nltk.corpus import wordnet
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import joblib
import numpy as np
from sklearn import feature_extraction
import collections
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import html2text


"""
函数说明：简单分词
Parameters:
     filename:数据文件
Returns:
     list_word_split：分词后的数据集列表
     category_labels: 文本标签列表
"""
def word_split(filename):
    read_data=pd.read_excel(filename)
    list_word_split=[]
    category_labels=[]
    for i in range(len(read_data)):
        row_data = read_data.iloc[i, 1]           # 读取单个漏洞描述文本
        list_row_data = list(jieba.cut(row_data)) # 对单个漏洞进行分词
        list_row_data=[x for x in list_row_data if x!=' '] #去除列表中的空格字符
        list_word_split.append(list_row_data)
 
        row_data_label=read_data.iloc[i,2]   #读取单个漏洞的类别标签
        category_labels.append(row_data_label) #将单个漏洞的类别标签加入列表
    return list_word_split, category_labels
 
 
"""
函数说明：词性还原
Parameters:
     list_words:数据列表
Returns:
     list_words_lemmatizer：词性还原后的数据集列表
"""
def word_lemmatizer(list_words):
    wordnet_lemmatizer = WordNetLemmatizer()
    list_words_lemmatizer = []
    for word_list in list_words:
        lemmatizer_word = []
        for i in word_list:
            lemmatizer_word.append(wordnet_lemmatizer.lemmatize(i))
        list_words_lemmatizer.append(lemmatizer_word)
    return list_words_lemmatizer
 
 
"""
函数说明：停用词过滤
Parameters:
     filename:停用词文件
     list_words_lemmatizer:词列表
Returns:
     list_filter_stopwords：停用词过滤后的词列表
"""
def stopwords_filter(filename,list_words_lemmatizer):
    list_filter_stopwords=[]  #声明一个停用词过滤后的词列表
    with open(filename,'r') as fr:
        stop_words=list(fr.read().split('\n')) #将停用词读取到列表里
        for i in range(len(list_words_lemmatizer)):
            word_list = []
            for j in list_words_lemmatizer[i]:
                if j not in stop_words:
                    word_list.append(j.lower()) #将词变为小写加入词列表
            list_filter_stopwords.append(word_list)
        return list_filter_stopwords


"""
函数说明：文本向量化，标签向量化   one-hot编码
Parameters:
     feature_words:特征词集
     doc_words:文本列表
     doc_category_labels:文本类别标签
Returns:
     docvec_list:文本向量列表
     labelvec_list:标签向量列表
"""
def words2vec(feature_words,doc_words,doc_category_labels):
    #文本列表转向量列表
    docvec_list=[]
    for words in doc_words:
        docvec = [0] * len(feature_words)
        for j in words:
            if j in feature_words:
                docvec[feature_words.index(j)]=1
        docvec_list.append(docvec)
    #标签列表转向量列表
    labelvec_list = []
    labelset=list(set(doc_category_labels))
    for label in doc_category_labels:
        doclabel = [0] * len(labelset)
        doclabel[labelset.index(label)]=1
        labelvec_list.append(doclabel)
    return docvec_list,labelvec_list


#同义词过滤
def get_syn(word_list):
    synonym_lists = []
    for word in word_list:
        syn = []
        for synonym in wordnet.synsets(word):
            for lemma in synonym.lemmas():
                syn.append(str(lemma.name()))
        synonym_lists.append(syn)
    return synonym_lists


def is_synonyms(synonym_lists,word):
    for index, synonym_set in enumerate(synonym_lists):
        if word in synonym_set:
            return index
    return -1


def synonyns_filtered(sample):
    synonym_lists = get_syn(sample)
    for i in range(len(sample)-1):
        word = sample[i]
        ind = is_synonyms(synonym_lists, word)
        if ind != -1:
            synonym_set = synonym_lists[ind]
            for m in range(i,len(sample)):
                if sample[m] in synonym_set:
                    sample[m] = word
    return sample


# 获取单词的词性
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def filter_homograph(train_document):
    for i in range(len(train_document)):
        train_document_new = train_document
        tagged_sent = pos_tag(train_document[i].split())
        wnl = WordNetLemmatizer()
        lemmas_sent = []
        for tag in tagged_sent:
            wordnet_pos = get_wordnet_pos(tag[1])
            lemmas_sent.append(wnl.lemmatize(tag[0],pos=wordnet_pos))
        lemmas_sent_new = ' '.join(lemmas_sent)
        train_document_new.loc[i] = lemmas_sent_new
    return train_document_new



if __name__=='__main__':
    # list_word_split, category_labels=word_split('./file/testdata.xls') #获得每条文本的分词列表和标签列表
    # print('分词成功')
    # list_words_lemmatizer=word_lemmatizer(list_word_split)  #词性还原
    # print('词性还原成功')
    # list_filter_stopwords=stopwords_filter('./file/stopwords.txt',list_words_lemmatizer) #获得停用词过滤后的列表
    # print("停用词过滤成功")

    # feature_words = nltk.text.TextCollection(list_filter_stopwords) 
    # feature_words = jieba.analyse.extract_tags(list_filter_stopwords, topK=5, withWeight=False, allowPOS=())

    # ticket = pd.read_excel(r'E:\Data\ICM_Kusto_Data_YTD_2021.xlsx',sheet_name='filter_s3_test')['IsOutage']
    # ticket = ticket.fillna(0)
    # ticket = pd.read_excel(r'E:\Data\ICM_Kusto_Data_YTD_2021.xlsx',sheet_name='filter_s3_test')['IncidentType']
    # ticket.loc[ticket=='CustomerReported']=0
    # ticket.loc[ticket=='LiveSite']=1

    data = pd.read_excel(r'E:\Data\ICM_Kusto_Data_YTD_2021.xlsx',sheet_name='filter_s3_test')
    data = data[data['IsOutage']==True]
    data = data.reset_index(drop=True)
    measure = 'Summary'
    train_document  = data[measure]
    train_document = train_document.fillna('-1')

    # html格式去除
    train_document = train_document.apply(lambda x: html2text.html2text(x).replace('\n', ' ').replace('\t', ' ').lower())
    train_document = train_document.apply(lambda x: re.compile(r'\\x09',re.S).sub(' ', x)) # 去除html'< >'格式
    # train_document = train_document.apply(lambda x: re.compile(r'<[^>]+>',re.S).sub(' ', x)) # 去除html'< >'格式
    # train_document = train_document.apply(lambda x: re.compile(r'&\w+;',re.S).sub(' ', x)) #去除html'& ;'格式

    # 日期时间格式代替
    # rex_date = r'((20\d{​2}​-(1[0-2]|0\d)-([0-2]\d|3[0-1]))|\d{​2}​\.(1[0-2]|0\d)\.([0-2]\d|3[0-1]))'
    # rex_time = r'(24:00:00|(2[0-3]|[0-1]\d|d):[0-5]\d:[0-5]\d)'
    # train_document = train_document.apply(lambda x: re.sub(rex_date, '', x))
    # train_document = train_document.apply(lambda x: re.sub(rex_time, '', x))
    rex_date = r"(\d{4}-\d{1,2}-\d{1,2}\s\d{1,2}:\d{1,2})"
    train_document = train_document.apply(lambda x: re.compile(rex_date,re.S).sub(' ', x))

    #词形还原
    train_document = filter_homograph(train_document)
    # pst = PorterStemmer()
    # train_document = train_document.apply(lambda y: ' '.join([pst.stem(x) for x in y.split()]))

    #同义词去除
    train_document = train_document.apply(lambda y: ' '.join(synonyns_filtered(y.split())))

    #feature--tfidf
    nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words('english')  
    f = open('stop_word.txt')
    stopwords_n = f.read().splitlines()
    stopwords = stopwords+stopwords_n
    tfidf_vectorizer = TfidfVectorizer(max_features=10, stop_words=stopwords) #
    tfidf_matrix = tfidf_vectorizer.fit_transform(train_document.values)
    dist_train = 1-cosine_similarity(tfidf_matrix)  

    # test_document = pd.read_excel('./file/testdata.xls',sheet_name='Sheet3')['reason']
    test_document = train_document
    test_document = test_document.fillna('-1')
    tfidf_matrix_test = tfidf_vectorizer.fit_transform(test_document.values)
    dist_test = 1-cosine_similarity(tfidf_matrix_test)   

    # vec_word = words2vec(feature_words, list_filter_stopwords, category_labels)
    # vec_word_n = np.array(feature_words)
    terms = tfidf_vectorizer.get_feature_names()

    # kmeans
    num_clusters = 2 #聚为四类，可根据需要修改
    km = KMeans(n_clusters=num_clusters, random_state=1)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()
    # joblib.dump(km, 'nlp_cluster.pkl') 

    # dist
    test_cluster = km.predict(tfidf_matrix_test)
    dist = []
    label = []
    for i in range(len(tfidf_matrix_test.toarray())):
        diff = km.cluster_centers_-tfidf_matrix_test.toarray()[i,:]
        dist_0 = np.sqrt(np.sum(diff**2,axis=-1))
        dist_min_0 = np.min(dist_0)
        if dist_min_0<2:    # 0.5:
            label_0 = np.where(dist_0==dist_min_0)[0]
        else:
            label_0 = num_clusters
        label = np.append(label,label_0)
        dist = np.append(dist,dist_min_0)
    
    # wordcloud
    word_1 = train_document.loc[np.where(np.array(clusters)==1)[0]]
    word_0 = train_document.loc[np.where(np.array(clusters)==0)[0]]
    wc_1 = WordCloud().generate(''.join(word_1.values))
    wc_0 = WordCloud().generate(''.join(word_0.values))

    # result
    print(np.mean(dist))
    print(terms)
    plt.imshow(wc_0, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    plt.imshow(wc_1, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    # result = pd.DataFrame()
    # result['ticket'] = ticket.values
    # result['label'] = label
    # print(result.value_count())

    # from sklearn.metrics import confusion_matrix, classification_report
    # C=confusion_matrix(ticket, label)
    # print(classification_report(ticket,label))
    # print(C)



# ------------
    # # LDA
    # #创建LDA对象
    # lda = LatentDirichletAllocation(n_components=4, learning_offset=50, random_state=0,n_jobs=-1,max_iter=1000)
    # #训练LDA模型，得到聚类结果
    # lda_result = lda.fit_transform(tfidf_matrix)
    # print(lda_result)


    # index1=list(range(tfidf_matrix.shape[0]))
    # vc=pd.Series(clusters,index=index1)
    # aa = collections.Counter(clusters)
    # v = pd.Series(aa)
    # v1=v.sort_values(ascending=False)

    # for n in range(3):
    #     vc1=vc[vc==v1.index[n]]
    #     vindex=list(vc1.index)
    #     kkp=pd.Series(terms)
    #     print('第',n,'类的前10个数据')        
    #     ffg=kkp[vindex][:10]
    #     ffg1=list(set(ffg))
    #     print(ffg1)
    
    # # 可视化
    # # 使用T-SNE算法，对权重进行降维，准确度比PCA算法高，但是耗时长
    # tsne = TSNE(n_components=2)
    # decomposition_data = tsne.fit_transform(tfidf_matrix.toarray())
    # x = []
    # y = []
    # for i in decomposition_data:
    #     x.append(i[0])
    #     y.append(i[1])
    # fig = plt.figure(figsize=(10, 10))
    # ax = plt.axes()
    # plt.scatter(x, y, c=km.labels_, marker="x")
    # plt.xticks(())
    # plt.yticks(())
    # plt.show()
    # # plt.savefig('./sample.png', aspect=1)
    
    # #wordcloud
    # # wc = WordCloud().generate(text=terms)
    # # plt.imshow(wc, interpolation='bilinear')
    # # plt.axis('off')
    # # plt.show()

    # # plot tree
    # from scipy.cluster.hierarchy import ward, dendrogram
    # linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances
    # fig, ax = plt.subplots(figsize=(15, 20)) # set size
    # ax = dendrogram(linkage_matrix, orientation="right", labels=label)
    # ax = dendrogram(linkage_matrix, truncate_mode='lastp',p=5,show_contracted=True, labels=label)
    # plt.tick_params(\
    #     axis= 'x',          # changes apply to the x-axis
    #     which='both',      # both major and minor ticks are affected
    #     bottom='off',      # ticks along the bottom edge are off
    #     top='off',         # ticks along the top edge are off
    #     labelbottom='off')
    # plt.tight_layout() #show plot with tight layout
    # #uncomment below to save figure
    # # print(end) #不抛出错误图显示不出来
    # # plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters
    # plt.show()




 

