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

# stopword
def stopwords_filter(stopwords,list_words_lemmatizer):
    list_filter_stopwords=[]  
    stop_words=list(stopwords) 
    for i in range(len(list_words_lemmatizer)):
        word_list = []
        for j in list_words_lemmatizer[i].split():
            if j not in stop_words:
                word_list.append(j.lower()) 
        list_filter_stopwords.append(' '.join(word_list))
    return pd.Series(list_filter_stopwords)


#同义词过滤
# def get_syn(word_list):
#     synonym_lists = []
#     for word in word_list:
#         syn = []
#         for synonym in wordnet.synsets(word):
#             for lemma in synonym.lemmas():
#                 syn.append(str(lemma.name()))
#         synonym_lists.append(syn)
#     return synonym_lists
def get_syn(word_list):
    tagged_sent = pos_tag(word_list)
    wnl = WordNetLemmatizer()
    synonym_lists = []
    for tag in tagged_sent:
        syn = []
        wordnet_pos = get_wordnet_pos(tag[1])
        for synonym in wordnet.synsets(tag[0],pos=wordnet_pos):
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
    data = pd.read_excel(r'E:\Data\ICM_Kusto_Data_YTD_2021.xlsx',sheet_name='filter_s3_test')
    data = data.fillna('-1')

    max_features = 3
    num_clusters = 3
    data = data[data['IsOutage']==True]
    state_isoutage = 'true'
    # data = data[data['IsOutage']=='-1']
    # state_isoutage = 'none'

    data = data.reset_index(drop=True)
    measure = 'Summary'
    train_document  = data[measure]

    # html格式去除
    train_document = train_document.apply(lambda x: html2text.html2text(x).replace('\n', ' ').replace('\t', ' ').lower())
    train_document = train_document.apply(lambda x: re.compile(r'\\x09',re.S).sub(' ', x)) # 去除html'< >'格式

    # 日期时间格式代替
    rex_date = r"(\d{4}-\d{1,2}-\d{1,2}\s\d{1,2}:\d{1,2})"
    train_document = train_document.apply(lambda x: re.compile(rex_date,re.S).sub(' ', x))

    # 去除‘:’
    train_document = train_document.apply(lambda x: re.compile(r':',re.S).sub(' ', x)) 

    #去除stopwords
    nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words('english')  
    f = open('stop_word.txt')
    stopwords_n = f.read().splitlines()
    stopwords = stopwords+stopwords_n
    train_document = stopwords_filter(stopwords,train_document)

    # 词形还原
    train_document = filter_homograph(train_document)
    # pst = PorterStemmer()
    # train_document = train_document.apply(lambda y: ' '.join([pst.stem(x) for x in y.split()]))

    #同义词去除
    train_document = train_document.apply(lambda y: ' '.join(synonyns_filtered(y.split())))

    #词形还原
    train_document = filter_homograph(train_document)

    #feature--tfidf
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stopwords) #
    tfidf_matrix = tfidf_vectorizer.fit_transform(train_document.values)
    dist_train = 1-cosine_similarity(tfidf_matrix)  

    # test_document = pd.read_excel('./file/testdata.xls',sheet_name='Sheet3')['reason']
    test_document = train_document
    tfidf_matrix_test = tfidf_vectorizer.fit_transform(test_document.values)
    dist_test = 1-cosine_similarity(tfidf_matrix_test)   

    terms = tfidf_vectorizer.get_feature_names()

    # kmeans
    num_clusters = num_clusters #聚为四类，可根据需要修改
    km = KMeans(n_clusters=num_clusters, random_state=1)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()

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
    for i in range(num_clusters):
        word = train_document.loc[np.where(np.array(clusters)==i)[0]]
        word_cloud = WordCloud(background_color = 'white',collocations=False).generate(''.join(word.values))
        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(r'E:\Data_analysis_figures\incident_correlation'+'/'
                        + state_isoutage+'_'+str(max_features)+'_'+str(num_clusters)+'_'+str(i)+'.jpeg')

    # result
    print(np.mean(dist))
    print(terms)

    id = data['IncidentId'] 
    file_savename = './file/icm_ticket/'+'/'+ state_isoutage+'_'+str(max_features)+'_'+str(num_clusters)+'_'+str(i)+'_'
    id.to_csv(file_savename+'id.csv')
    np.savetxt(file_savename+'dist_test.csv',dist_test,delimiter=',')
    np.savetxt(file_savename+'tfidf.csv',tfidf_matrix.toarray(),delimiter=',')

    # cluster_figure
    from sklearn.manifold import MDS
    from sklearn.decomposition import PCA
    # MDS降维算法
    MDS()
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(dist_test)  # shape (n_components, n_samples)
    ## pca降维算法
    # pca = PCA(n_components=2)
    # pos = pca.fit_transform(dist_test)
    # pos = pca.fit_transform(tfidf_matrix.toarray())
    xs, ys = pos[:, 0], pos[:, 1]

    #set up colors per clusters using a dict
    cluster_colors = {0: '#FF0000', 1: '#d95f02', 2: '#7570b3'} #, 3: '#FFFF00', 4: '#66a61e'}
    #set up cluster names using a dict
    cluster_names = {0: 'label=0', 
                    1: 'label=1', 
                    2: 'label=2'} #, 
                   # 3: 'label=3',
                   # 4: 'label=4'}
    #create data frame that has the result of the cluster numbers and titles
    df = pd.DataFrame(dict(x=xs, y=ys, label=clusters)) 

    #group by cluster
    groups = df.groupby('label')
    # set up plot
    fig, ax = plt.subplots(figsize=(17, 9)) # set size
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    #iterate through groups to layer the plot
    #note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
                label=cluster_names[name], color=cluster_colors[name], 
                mec='none')
        ax.set_aspect('auto')
        ax.tick_params(\
            axis= 'x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')
        ax.tick_params(\
            axis= 'y',         # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            left='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelleft='off')    
    ax.legend(numpoints=1)  #show legend with only 1 point
    #add label in x,y position with the label as the film title
    for i in range(len(df)):
        #与loc不同的之处是，.iloc 是根据行数与列数来索引的
        ax.text(df.loc[i]['x'], df.loc[i]['y'], df.loc[i]['label'].astype(int), size=8)  
    plt.show()
    plt.savefig(r'E:\Data_analysis_figures\incident_correlation'+'/'
                         + state_isoutage+'_'+str(max_features)+'_'+str(num_clusters)+'_'+'cluster'+'.jpeg')

    # plt.show() #show the plot



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
