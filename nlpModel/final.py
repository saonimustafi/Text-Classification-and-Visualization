#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import tarfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from lime import lime_text
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
from matplotlib import pyplot as plt
import re
import pandas as pd
from collections import defaultdict


def train_classifier(X, y):
    cls=LogisticRegression(random_state=0, max_iter=10000)
    cls.fit(X, y)
    return cls

def evaluate(X, yt, cls, name='data'):
    yp = cls.predict(X)
    acc = metrics.accuracy_score(yt, yp)
    print("  Accuracy on %s  is: %s" % (name, acc))

def read_files(tarfname):   
    tar = tarfile.open(tarfname, "r:gz")
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name
            
            
    class Data: pass
    sentiment = Data()
    print("-- train data")
    sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)
    print(len(sentiment.train_data))

    print("-- dev data")
    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    print(len(sentiment.dev_data))
    print("-- transforming data and labels")
    """" REPLACE COUNTS BY TFIDF OF WORDS"""
    sentiment.tfidf_vect = TfidfVectorizer(max_df=0.8,sublinear_tf=True, use_idf=True)
    sentiment.trainX = sentiment.tfidf_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.tfidf_vect.transform(sentiment.dev_data)
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    tar.close()
    return sentiment

def write_pred_kaggle_file(unlabeled, cls, outfname, sentiment):
    yp = cls.predict(unlabeled.X)
    labels = sentiment.le.inverse_transform(yp)
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    for i in range(len(unlabeled.data)):
        f.write(str(i+1))
        f.write(",")
        f.write(labels[i])
        f.write("\n")
    f.close()
    
def read_unlabeled(tarfname, sentiment):
    tar = tarfile.open(tarfname, "r:gz")
    class Data: pass
    unlabeled = Data()
    unlabeled.data = []
    unlabeledname = "unlabeled.tsv"
    for member in tar.getmembers():
        if 'unlabeled.tsv' in member.name:
            unlabeledname = member.name
            
    print(unlabeledname)
    tf = tar.extractfile(unlabeledname)
    for line in tf:
        line = line.decode("utf-8")
        text = line.strip()
        unlabeled.data.append(text)
        
            
    unlabeled.X = sentiment.tfidf_vect.transform(unlabeled.data)
    print(unlabeled.X.shape)
    tar.close()
    return unlabeled

def read_tsv(tar, fname):
    member = tar.getmember(fname)
    print(member.name)
    tf = tar.extractfile(member)
    data = []
    labels = []
    for line in tf:
        line = line.decode("utf-8")
        (label,text) = line.strip().split("\t")
        labels.append(label)
        data.append(text)
    return data, labels


def prediction(txt, sentiment, logistic, num_features):
    ##LIME
    c = make_pipeline(sentiment.tfidf_vect, logistic)
    class_names = ['NEGATIVE', 'POSITIVE']
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(txt, c.predict_proba, num_features=num_features)
    output = "static/outputs/output.html"
    exp.save_to_file(output)
    exp.as_pyplot_figure(label=1)
    plt.savefig('static/outputs/lime_explanation_graph.png')

    # LOGISTIC REGRESSION
    list_of_words = re.sub("[^\w]", " ", txt).split()
    words_with_weights = defaultdict()
    for word in list_of_words:
        feats = sentiment.tfidf_vect.get_feature_names()
        coefs = logistic.coef_[0]
        if word in feats:
            ind = feats.index(word)
            words_with_weights[word] = coefs[ind]

    data = pd.DataFrame.from_dict(words_with_weights, orient='index')
    data[0].plot(kind='barh', color=(data[0] > 0).map({True: 'g', False: 'r'}))
    plt.savefig('static/outputs/log_explanation_graph.png')





def getResults(txt,num_of_features):
    print ("TEXT = " + txt)
    print("Reading data")
    tarfname = "static/data/sentiment.tar.gz"
    sentiment = read_files(tarfname)
    print("\nTraining classifier")

    cls = train_classifier(sentiment.trainX, sentiment.trainy)
    print("\nEvaluating")
    evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
    evaluate(sentiment.devX, sentiment.devy, cls, 'dev')
    print("\nReading unlabeled data")
    unlabeled = read_unlabeled(tarfname, sentiment)

    """" HYPERPARAMTER TUNING"""

    C = [0.01, 0.1, 1.0, 10.0, 100.0, 200.0, 1000.0, 10000.0]
    penalty = ['l1', 'l2']
    max_acc = 0
    final_penalty = 'l1'
    final_C = 0.1
    for i in C:
        for p in penalty:
            logistic = LogisticRegression(penalty=p, C=i, random_state=0, max_iter=10000)
            logistic.fit(sentiment.trainX, sentiment.trainy)
            yp = logistic.predict(sentiment.devX)
            acc = metrics.accuracy_score(sentiment.devy, yp)
            print("  Accuracy on penalty= %s C= %s is: %s" % (p, i, acc))
            if acc > max_acc:
                max_acc = acc
                final_penalty = p
                final_C = i

    # FINAL MODEL PERFORMACE ON TRAIN AND DEV SET
    logistic = LogisticRegression(penalty='l2', C=10.0, random_state=0, max_iter=10000)
    logistic.fit(sentiment.trainX, sentiment.trainy)
    yp = logistic.predict(sentiment.trainX)
    acc = metrics.accuracy_score(sentiment.trainy, yp)
    print("  FINAL Accuracy on TRAIN SET penalty= l2 C= 10 is: %s" % (acc))
    yp = logistic.predict(sentiment.devX)
    acc = metrics.accuracy_score(sentiment.devy, yp)
    print("  FINAL Accuracy on DEV SET penalty= l2 C= 10 is: %s" % (acc))

    prediction(txt, sentiment, logistic, num_of_features)






