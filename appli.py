from flask import Flask
from flask import request
from flask import render_template

from fonction_ import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
#import nltk
#from nltk import word_tokenize

#def tokenize_tags(x):
#    wt = word_tokenize(x)
#    return wt



#SVC_ = joblib.load('svc_3.pkl')
TfidfVec = joblib.load('vectorizer3.pkl')
features = np.array(TfidfVec.get_feature_names())  #feature_array
print('features shape',features.shape)
all_tags = joblib.load('tags_output3.pkl')
NMF_ = joblib.load('nmf.pkl')

app = Flask(__name__)


@app.route('/')
def home():
    print('hello im in home')
    return render_template("home.html")

@app.route('/', methods=['POST'])
def text_box():

    question = request.form['text']
    print("question is" , question)

    all_data = recommander_f(question,all_tags, TfidfVec,NMF_)#, SVC_, NMF_)
    print(all_data)

    if (len(all_data) == 0):
        processed_text = 'Please ask another question , no tags were found'
        return render_template("tag_reco_error.html", message=processed_text )
    else:
        processed_text = all_data
        return render_template("tag_reco.html", message="We propose 10 probable tags", message2 = "We propose also 10 possible tags",
                               supervised_tags= all_data[0:10], unsupervised_tags= all_data[10:20])
if __name__ == '__main__':
    app.run()
