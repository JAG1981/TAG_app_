from flask import Flask
from flask import request
from flask import render_template
import pandas as pd
import nltk
from fonction_ import *

from nltk import word_tokenize
def tokenize_tags(x):
    wt = word_tokenize(x)
    return wt

df1 = pd.read_csv('tags_and_titles_.csv', sep='\t', encoding='utf-8')
df1['tags_F'] = df1['tags_1'].map(tokenize_tags)
df2 = pd.read_csv("features.csv", sep='\t', encoding='utf-8')
SVC_ = joblib.load('svc_1.pkl')

app = Flask(__name__)

#df1 = pd.read_csv('tags_and_titles.csv', sep='\t', encoding='utf-8')
#print('file 1 has been read', df1.shape)

@app.route('/')
def home():
    print('hello im in home')
    return render_template("home.html")

@app.route('/', methods=['POST'])
def text_box():

    question = request.form['text']
    print("question is" , question)



    #print('file 1 has been read', df1.shape)
    #print('file 1 columns', df1.columns)
    #toto = df1['tags_F'].iloc[0]
    #print(toto)
    #print(type(toto))



    tag_set_F = set()
    total_sample = 30000
    for i in range(0, total_sample):
        tag = df1['tags_F'].iloc[i]
        for j in range(0, len(tag)):
            tag_set_F.add(tag[j])
    all_tags = list(tag_set_F)
    print('all_tags[0]',all_tags[0] )
    titles = list(df1['title_5'])



    features = list(df2['features'])
    print('file 2 has been read', df2.shape)

    all_data = recommander_(question,all_tags, features,titles, SVC_)
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
