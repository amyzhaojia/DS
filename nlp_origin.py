import pandas as pd
import numpy as np
import joblib
import nltk
import re
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import html2text 


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
    ticket = pd.read_excel(r'E:\Data\ICM_Kusto_Data_YTD_2021.xlsx',sheet_name='filter_s3_test')['IncidentType']
    ticket.loc[ticket=='CustomerReported']=0
    ticket.loc[ticket=='LiveSite']=1
    measure = 'Summary'

    nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words('english')  
    f = open('stop_word.txt')
    stopwords_n = f.read().splitlines()
    stopwords = stopwords+stopwords_n
    tfidf_vectorizer = TfidfVectorizer(max_features=3, stop_words=stopwords) #

    train_document  = pd.read_excel(r'E:\Data\ICM_Kusto_Data_YTD_2021.xlsx',sheet_name='filter_s3_test')[measure]
    train_document = train_document.fillna('-1')

    # html格式去除
    train_document = train_document.apply(lambda x: html2text.html2text(x).replace('\n', ' ').replace('\t', ' ').lower())
    train_document = train_document.apply(lambda x: re.compile(r'\\x09',re.S).sub(' ', x)) # 去除html'< >'格式

    # 日期时间格式代替
    rex_date = r"(\d{4}-\d{1,2}-\d{1,2}\s\d{1,2}:\d{1,2})"
    train_document = train_document.apply(lambda x: re.compile(rex_date,re.S).sub(' ', x))

    #词形还原
    train_document = filter_homograph(train_document)
    # pst = PorterStemmer()
    # train_document = train_document.apply(lambda y: ' '.join([pst.stem(x) for x in y.split()]))

    #同义词去除
    train_document = train_document.apply(lambda y: ' '.join(synonyns_filtered(y.split())))

    #feature--tfidf
    tfidf_matrix = tfidf_vectorizer.fit_transform(train_document.values)
    dist = 1-cosine_similarity(tfidf_matrix)  
    terms = tfidf_vectorizer.get_feature_names()

    # kmeans
    num_clusters = 2 #聚为四类，可根据需要修改
    km = KMeans(n_clusters=num_clusters, random_state=1)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()

    # test_document = pd.read_excel('./file/testdata.xls',sheet_name='Sheet3')['reason']
    test_document = train_document
    test_document = test_document.fillna('-1')
    tfidf_matrix_test = tfidf_vectorizer.fit_transform(test_document.values)
    dist_test = 1-cosine_similarity(tfidf_matrix_test)   
    terms = tfidf_vectorizer.get_feature_names()

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



# # 获取数据
# train_document  = pd.read_excel('./file/testdata.xls',sheet_name='Sheet2')['reason']
# test_document = pd.read_excel('./file/testdata.xls',sheet_name='Sheet3')['reason']
# test_document = test_document.fillna('-1')
# # 获取stopwords
# nltk.download('stopwords')
# stopwords = nltk.corpus.stopwords.words('english')  
# # 提取特征
# tfidf_vectorizer = TfidfVectorizer(max_features=3,stop_words=stopwords)
# tfidf_matrix = tfidf_vectorizer.fit_transform(train_document.values)
# tfidf_matrix_test = tfidf_vectorizer.fit_transform(test_document.values)
# # print(tfidf_matrix.shape)   

# terms = tfidf_vectorizer.get_feature_names()
# dist = 1-cosine_similarity(tfidf_matrix)  
# dist_test = 1-cosine_similarity(tfidf_matrix_test)  

# # kmeans聚类
# # train
# num_clusters = 4 #聚为四类，可根据需要修改
# km = KMeans(n_clusters=num_clusters, random_state=1)
# km.fit(tfidf_matrix)
# clusters = km.labels_.tolist()
# joblib.dump(km, 'nlp_cluster.pkl') 
# # test
# test_cluster = km.predict(tfidf_matrix_test)
# dist = []
# label = []
# for i in range(len(tfidf_matrix_test.toarray())):
#     diff = km.cluster_centers_-tfidf_matrix_test.toarray()[i,:]
#     dist_0 = np.sqrt(np.sum(diff**2,axis=-1))
#     dist_min_0 = np.min(dist_0)
#     if dist_min_0<0.5:
#         label_0 = np.where(dist_0==dist_min_0)[0]
#     else:
#         label_0 = num_clusters
#     label = np.append(label,label_0)
#     dist = np.append(dist,dist_min_0)
# print(dist)
