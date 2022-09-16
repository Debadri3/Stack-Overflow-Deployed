from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import re 

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier


import scipy
from scipy.sparse import hstack

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

def striphtml(data):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(data))
    return cleantext
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# load the model from disk
filename = 'one-rest.pkl'
clf = pickle.load(open(filename, 'rb'))
x_vectorizer=pickle.load(open('tf-vectorizer.pkl','rb'))
y_vectorizer=pickle.load(open('bow-vectorizer.pkl','rb'))

application = Flask(__name__)

@application.route('/')
def home():
	return render_template('home.html')

@application.route('/predict',methods=['POST','GET'])
def predict():
    

 if request.method == 'POST':
  
  title = request.form['title']
  data1 = [title]

  question=request.form['question'] 
  data2=[question]

  code = str(re.findall(r'<code>(.*?)</code>', question, flags=re.DOTALL))

  question=re.sub('<code>(.*?)</code>', '', question, flags=re.MULTILINE|re.DOTALL)
  question=striphtml(question.encode('utf-8'))

  title=title.encode('utf-8')

  question=str(title)+" "+str(question)
  question=re.sub(r'[^A-Za-z]+',' ',question)
  words=word_tokenize(str(question.lower()))

  #Removing all single letter and stopwords from question except for the letter 'c'
  question=' '.join(str(lemmatizer.lemmatize(j)) for j in words if j not in stop_words and (len(j)!=1 or j=='c'))
  question=[question]

  x_transformed=x_vectorizer.transform(question)
  x_transformed=normalize(x_transformed,axis=0)
        
         
  my_prediction = clf.predict(x_transformed)
  
  tag_list=[]
  for col in range (0,500):
    if my_prediction.toarray()[0][col]==1:
      tag_list.append(y_vectorizer.get_feature_names()[col])
  s = ",".join(tag_list)
  
 return render_template('home.html',prediction = s)


if __name__ == '__main__':
	application.run(debug=True)
