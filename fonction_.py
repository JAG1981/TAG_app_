import numpy as np
import warnings
#import nltk
#from nltk import word_tokenize
#from nltk.corpus import stopwords
import sklearn
# for feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer
# algorithms
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.feature_extraction import stop_words


def recommander_f(q, alltags, TfidfVec_, SVC_prob, NMF):

    warnings.filterwarnings('ignore')
    tag_nb = 10

    question = q
    all_tags = alltags
    features = np.array(TfidfVec_.get_feature_names())
    #tok_question = word_tokenize(question.lower())
    vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    to = []
    to.append(question.lower())
    tokenized = vectorizer.fit_transform(to)
    # vectorizer.build_tokenizer()
    result = vectorizer.inverse_transform(tokenized)
    tok_question = list(result[0])
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