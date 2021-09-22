import pandas as pd
import numpy as np
import nltk
# nltk.download('wordnet')
# nltk.download('punkt')
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
from sklearn.metrics.pairwise import cosine_similarity

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems
 
 
def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


if __name__=='__main__':
    test_document = pd.read_excel('./file/testdata.xls',sheet_name='Sheet3')['reason']
    test_document = test_document.fillna('-1')
    nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words('english') 
    tfidf_vectorizer = TfidfVectorizer(max_features=3,stop_words=stopwords)
    tfidf_matrix_test = tfidf_vectorizer.fit_transform(test_document.values) 
    terms = tfidf_vectorizer.get_feature_names()
    num_clusters = 4 #聚为四类，可根据需要修改
    km = joblib.load('nlp_cluster.pkl')
    # clusters = km.labels_.tolist()  
    test_cluster = km.predict(tfidf_matrix_test)
    dist_test = 1-cosine_similarity(tfidf_matrix_test)  
    dist = []
    label = []
    for i in range(len(tfidf_matrix_test.toarray())):
        diff = km.cluster_centers_-tfidf_matrix_test.toarray()[i,:]
        dist_0 = np.sqrt(np.sum(diff**2,axis=-1))
        dist_min_0 = np.min(dist_0)
        if dist_min_0<0.5:
            label_0 = np.where(dist_0==dist_min_0)[0]
        else:
            label_0 = num_clusters
        label = np.append(label,label_0)
        dist = np.append(dist,dist_min_0)
    # print(dist)
    clusters = label

    totalvocab_stemmed = []
    totalvocab_tokenized = []
    for i in test_document:
        allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
        totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list   
        allwords_tokenized = tokenize_only(i)
        totalvocab_tokenized.extend(allwords_tokenized)
    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
    print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')  

    
    films = {'test_document': test_document, 'cluster': clusters}
    frame = pd.DataFrame(films, index = [clusters] , columns = ['cluster'])
    print(frame['cluster'].value_counts()) #

    # print("Top terms per cluster:")
    # #sort cluster centers by proximity to centroid
    # order_centroids = km.cluster_centers_.argsort()[:, ::-1] 
    # for i in range(num_clusters):
    #     print("Cluster %d words: " %i, end='') #%d功能是转成有符号十进制数 #end=''让打印不要换行
    #     for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
    #         #b'...' is an encoded byte string. the unicode.encode() method outputs a byte string that needs to be converted back to a string with .decode()
    #         print('%s' %vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=', ')
    #     print() #add whitespace

    import os  # for os.path.basename
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from sklearn.manifold import MDS
    MDS()
    # convert two components as we're plotting points in a two-dimensional plane
    # "precomputed" because we provide a distance matrix
    # we will also specify `random_state` so the plot is reproducible.
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(dist_test)  # shape (n_components, n_samples)
    xs, ys = pos[:, 0], pos[:, 1]
    print(xs, ys)

    #set up colors per clusters using a dict
    cluster_colors = {0: '#FF0000', 1: '#d95f02', 2: '#7570b3', 3: '#FFFF00', 4: '#66a61e'}
    #set up cluster names using a dict
    cluster_names = {0: 'No new data', 
                    1: 'Timed out', 
                    2: 'Too many requests', 
                    3: 'phoneMonitor failed',
                    4: 'Others'}

    #create data frame that has the result of the MDS plus the cluster numbers and titles
    df = pd.DataFrame(dict(x=xs, y=ys, label=label)) 
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
    plt.show() #show the plot
    #uncomment the below to save the plot if need be
    #plt.savefig('clusters_small_noaxes.png', dpi=200)
    #plt.close()

    #create data frame that has the result of the MDS plus the cluster numbers and titles
    df = pd.DataFrame(dict(x=xs, y=ys, label=clusters)) 
    #group by cluster
    groups = df.groupby('label')
    #define custom css to format the font and to remove the axis labeling
    css = """
    text.mpld3-text, div.mpld3-tooltip {
    font-family:Arial, Helvetica, sans-serif;
    }
    g.mpld3-xaxis, g.mpld3-yaxis {
    display: none; }
    svg.mpld3-figure {
    margin-left: -200px;}
    """
    # Plot 
    fig, ax = plt.subplots(figsize=(14,6)) #set plot size
    ax.margins(0.03) # Optional, just adds 5% padding to the autoscaling
    #iterate through groups to layer the plot
    #note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
    for name, group in groups:
        points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=18, 
                        label=cluster_names[name], mec='none', 
                        color=cluster_colors[name])
        ax.set_aspect('auto')
        labels = [i for i in group.label]
        #set tooltip using points, labels and the already defined 'css'
        tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels,
                                        voffset=10, hoffset=10, css=css)
        #connect tooltip to fig
        mpld3.plugins.connect(fig, tooltip, TopToolbar())    
        #set tick marks as blank
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        #set axis as blank
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    ax.legend(numpoints=1) #show legend with only one dot
    print(end) #不抛出错误图显示不出来
    mpld3.display() #show the plot
    #uncomment the below to export to html
    #html = mpld3.fig_to_html(fig)
    #print(html)

    from scipy.cluster.hierarchy import ward, dendrogram
    linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances
    fig, ax = plt.subplots(figsize=(15, 20)) # set size
    ax = dendrogram(linkage_matrix, orientation="right")
    plt.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    plt.tight_layout() #show plot with tight layout
    #uncomment below to save figure
    # print(end) #不抛出错误图显示不出来
    plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters
    plt.close()

