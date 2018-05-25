import numpy as np
import random
import pandas as pd
import warnings
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords


import sklearn
# for feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer
# algorithms
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.feature_extraction import stop_words



class Reco_movie(object):
    def __init__(self):
        self.movie_reco = []
        self.movie_id = []
        self.intro = []
        self.reco = ""


def recommander_f(q, alltags, TfidfVec_, SVC_prob, NMF):

    warnings.filterwarnings('ignore')
    tag_nb = 10



    question = q
    all_tags = alltags
    features = np.array(TfidfVec_.get_feature_names())
    tok_question = word_tokenize(question.lower())
    retained_words = ''
    for w in tok_question:
        if (w in features):
            retained_words += w + ' '

    print('words',retained_words)

    ## partie supervisée

    X_tfidf = TfidfVec_.transform([retained_words])

    prob_svc = SVC_prob.predict_proba(X_tfidf)#[-1])
    print('loading')
    tags_output = np.asarray(all_tags)
    print(tags_output[0:100])
    ind_svc = np.argpartition(prob_svc[0], range(len(prob_svc[0])))[-tag_nb:]
    print(ind_svc)
    print('tags output',len(tags_output))
    print("\nSVC proba", list(reversed(prob_svc[0][ind_svc])))
    svc_word = tags_output[ind_svc]
    supervised = list(reversed(svc_word))


    ## partie non supervisée


    print("unsupervised")
    post_topic_KL = NMF.transform(X_tfidf)
    print("ok transform")
    topic_most_prKL = post_topic_KL[0].argsort()[-2:]
    print('ok sort')
    witKL = []
    for k in topic_most_prKL:
        print(k)
        topic = NMF.components_[k]
        witKL += [features[i] for i in topic.argsort()[:-tag_nb//2 - 1:-1]]

    inter_set_KL = set(witKL)
    unsupervised = list(inter_set_KL)
    print(unsupervised )

    output = supervised + unsupervised
    print(output)

    return (output)




def recommander_(q, alltags, features_,titles_, SVC_prob):

    warnings.filterwarnings('ignore')

    all_englis_stop_word = set()
    all_english_stop_word = set(stop_words.ENGLISH_STOP_WORDS).union(set(stopwords.words('english')))
    TfidfVec = TfidfVectorizer(stop_words=all_english_stop_word, max_df=0.85, sublinear_tf=True, lowercase=True)
    train_sample = 29000
    total_sample = 30000
    tag_nb = 20

    question = q
    all_tags = alltags
    features = features_
    tok_question = word_tokenize(question.lower())
    retained_words = ''
    for w in tok_question:
        if (w in features):
            retained_words += w + ' '
    X_total = []
    train = titles_[0:train_sample]
    test = titles_[train_sample:total_sample]
    X_total  = np.concatenate((train, test) )
    X_T = np.append(X_total,np.array(retained_words))
    X_tfidf = TfidfVec.fit_transform(X_T.astype('U'))
    prob_svc_question = SVC_prob.predict_proba(X_tfidf[-1])
    print('loading')
    tags_output = np.asarray(all_tags)
    ind_svc = np.argpartition(prob_svc_question[0], -tag_nb)[-tag_nb:]
    print('tags output',len(tags_output))
    print(ind_svc )
    #print("\nSVC proba", prob_svc_question[0][ind_svc])
    #print("SVC feature", tags_output[ind_svc])
    return(list(tags_output[ind_svc]))

def recommander(title, data):
    result = data

    list_movie_ = list(result['movie_title'])
    list_movie_reco = list()
    list_intro = list()
    list_id_movie = list()
    list_result = list()
    movie_idea_nb= 5
    print_title= False
    result_list = list()
    is_movie =  False
    index_movie = -1
    string2=""

    for i in range(0, len(list_movie_)):
        movie_name = list_movie_[i]
        if(title.lower() in movie_name.lower() ) :
            result_list.append(list_movie_[i])
            index_movie = i
            if(len(result_list)>1):
                string = 'SVP soyez plus explicite avec le titre du film'
                #print(string)
                list_intro.append(0)
                list_intro.append(string)
                is_movie = True
                break

    if(len(result_list)==1):
        string = "Le film de votre selection est " + result_list[0]
        #string = string.replace(u'\xa0', u'').encode('utf-8')
        list_intro.append(string)

        is_movie = True
        cat_movie = result[result['movie_title'] == list_movie_[index_movie]].loc[:,['Category']]['Category']

        list_similar_movie = list(result[result['Category'] == int(cat_movie)].loc[:,['movie_title']]['movie_title'])
        rand_items = random.sample(list_similar_movie, movie_idea_nb)
        string2 = " Nous vous recommandons alors les tags suivants:"

        for i in range(0,len(rand_items)):
            other_movie= rand_items[i]
            list_id_movie.append(result[result['movie_title'] == rand_items[i]].index[0])
            list_movie_reco.append(other_movie)

    if((not is_movie)):
        list_intro.append(0)
        string = "Il n'y aucun film qui contient le mot " + title
        list_intro.append(string)

    my_reco = Reco_movie()
    my_reco.reco = string2
    my_reco.movie_reco = list(list_movie_reco)
    my_reco.movie_id = list(list_id_movie)
    my_reco.intro = list(list_intro)
    return my_reco