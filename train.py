from load_data import *
from preprocess import *
import nltk
from nltk.corpus import wordnet 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd 
from sklearn import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

import pickle
from numpy import asarray
from numpy import save

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words)) for doc in sents])
    return targets, regressors

def training():
    data=pd.DataFrame()
    data=load_data_phishing(data)
    data_enron=pd.DataFrame()
    data_enron=load_data_enron(data_enron, 0)
    data=pd.concat([data,data_enron])

    cnt_pro = data['Class'].value_counts()

    plt.figure(figsize=(10,5))
    sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)
    plt.ylabel('Количество писем', fontsize=12)
    plt.xlabel('Вредоносность', fontsize=12)
    plt.xticks(rotation=90)
    plt.show();
    plt.savefig("data")

    data['Text'] = data['Text'].apply(cleanText)
    data['Text']=data['Text'].apply(lambda x: remove_stopwords(x))
    data['Text']=data['Text'].apply(lambda x: ' '.join([WordNetLemmatizer().lemmatize(word,pos=wordnet.VERB) for word in x.split()]))

    from sklearn.model_selection import train_test_split
    train, test = train_test_split(data, test_size=0.3, random_state=42)

    # ## Training model Doc2Vec

    tqdm.pandas(desc="progress-bar")

    train_tagged = train.apply(
        lambda r: TaggedDocument(words=tokenize_text(r['Text']), tags=[r.Class]), axis=1)
    test_tagged = test.apply(
        lambda r: TaggedDocument(words=tokenize_text(r['Text']), tags=[r.Class]), axis=1)

    train_tagged.shape

    model_dbow = Doc2Vec()
    model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

    for epoch in range(30):
        model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha

    y_train, X_train = vec_for_learning(model_dbow, train_tagged)
    y_test, X_test = vec_for_learning(model_dbow, test_tagged)

    # ## Classification

    model_pip=[]
    model_pip.append(('Logistic Regression',LogisticRegression()))
    model_pip.append(('Random Forest',RandomForestClassifier()))
    model_pip.append(('XGBoost',XGBClassifier()))
    model_pip.append(('SVM',SVC(probability=True)))
    model_pip.append(('KNN',KNeighborsClassifier()))
    model_pip.append(('Naive Bayes',GaussianNB()))
    model_pip.append(('Decision Tree',DecisionTreeClassifier()))
    model_pip.append(('MLP',MLPClassifier()))
    model_pip.append(('AdaBoost',AdaBoostClassifier()))
    model_pip.append(('Gradient Boosting',GradientBoostingClassifier()))

    return X_test, y_test, model_pip, X_train, y_train

if __name__ == "__main__":
    X_test, y_test, model_pip, X_train, y_train = training()

    data_X_test = asarray(X_test)
    data_y_test = asarray(y_test)
    save("data_X_test.npy", data_X_test)
    save("data_y_test.npy", data_y_test)

    # Write the model to a file
    if not os.path.isdir("dumped_models/classification/"):
        os.mkdir("dumped_models/classification")

    for classifier in model_pip:
        classifier[1].fit(X_train,y_train)
        pickle.dump(classifier[1],open('dumped_models/classification/'+classifier[0]+'.pkl','wb'))    