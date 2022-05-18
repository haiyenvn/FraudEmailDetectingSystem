# %%
import pandas as pd 

# %%
data=pd.DataFrame()
import os
for i in os.listdir('./phishing/'):
    dataa_new=pd.read_csv('./phishing/'+i)
    lable=dataa_new['content']
    nhan=[1]*len(lable)
    data=data.append(pd.DataFrame({'Text':lable,'Class':nhan}))

data=data.dropna()
print(data.shape)

# %%
data_enron=pd.DataFrame()
for i in os.listdir('./enron/'):
    dataa_new=pd.read_csv('./enron/'+i)

    lable=dataa_new['content']
    nhan=[0]*len(lable)
    data_enron=data_enron.append(pd.DataFrame({'Text':lable,'Class':nhan}))
    data_enron=data_enron.dropna()
data=pd.concat([data,data_enron])
# data=data.dropna()
print(data.head)

# %%
from sklearn.utils import shuffle
data=shuffle(data,random_state=42)
data

# %%
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = nltk.word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return ' '.join(filtered_sentence)

# %%
from nltk.corpus import wordnet 
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
data['Text']=data['Text'].apply(lambda x: remove_stopwords(x))
data['Text']=data['Text'].apply(lambda x: ' '.join([WordNetLemmatizer().lemmatize(word,pos=wordnet.VERB) for word in x.split()]))

# %%
from matplotlib import pyplot as plt
import seaborn as sns

# %%
cnt_pro = data['Class'].value_counts()

plt.figure(figsize=(10,5))
sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)
plt.ylabel('Количество писем', fontsize=12)
plt.xlabel('Вредоносность', fontsize=12)
plt.xticks(rotation=90)
plt.show();

# %%
from bs4 import BeautifulSoup
import re
def cleanText(text):
    text = BeautifulSoup(text, "html.parser").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text
data['Text'] = data['Text'].apply(cleanText)

# %%
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.3, random_state=42)

# %%
import nltk
from nltk.corpus import stopwords
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

# %%
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

# %%
train_tagged = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['Text']), tags=[r.Class]), axis=1)
test_tagged = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['Text']), tags=[r.Class]), axis=1)

# %%
train_tagged.shape

# %%
import multiprocessing

cores = multiprocessing.cpu_count()

# %%
model_dbow = Doc2Vec()
model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

# %%
from sklearn import utils

# %%
%%time
for epoch in range(30):
    model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha

# %%
def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words)) for doc in sents])
    return targets, regressors

# %%
y_train, X_train = vec_for_learning(model_dbow, train_tagged)
y_test, X_test = vec_for_learning(model_dbow, test_tagged)

# %%
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


# %%
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# %%
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


# %%
acc=[]
f1=[]
precision=[]
recall=[]
roc_auc=[]
conf_mat=[]
cl=[]

# %%
import pickle
for classifier in model_pip:
    classifier[1].fit(X_train,y_train)
    y_pred=classifier[1].predict(X_test)
    acc.append(accuracy_score(y_test,y_pred))
    f1.append(f1_score(y_test,y_pred))
    precision.append(precision_score(y_test,y_pred))
    recall.append(recall_score(y_test,y_pred))
    conf_mat.append(confusion_matrix(y_test,y_pred))
    cl.append(classifier[1])
    pickle.dump(classifier[1],open('model/'+classifier[0]+'.pkl','wb'))    

# %%
plt.rcParams.update({'font.size': 22})

# %%
fig=plt.figure(figsize=(20,50))

for i in range(0,len(conf_mat)):
    cm_con=conf_mat[i]
    model=model_pip[i][0]
    sub_fig_title=fig.add_subplot(5,2,i+1).set_title(model)
    plot_map=sns.heatmap(cm_con,annot=True,cmap='Greens_r',fmt='g')
    plot_map.set_xlabel('Predicted_Values')
    plot_map.set_ylabel('Actual_Values')

# %%
result=pd.DataFrame({'Model': model_pip, 'Accuracy': acc, 'F1-score': f1, 'Precision': precision, 'Recall': recall})
result

# %%
fig=plt.figure(figsize=(20,50))

for i in range(0,len(cl)):
    c=cl[i]
    model=model_pip[i][0]
    y_pred_proba=c.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:,1])
    auc=roc_auc_score(y_test,y_pred_proba[:,1])
    sub_fig_title=fig.add_subplot(5,2,i+1).set_title(model)
    plot_map=sns.lineplot(x=fpr,y=tpr,label='AUC='+str(auc))
    plot_map.set_xlabel('False Positive Rate')
    plot_map.set_ylabel('True Positive Rate')


# %%
fig=plt.figure(figsize=(20,10))

for i in range(0,len(cl)):
    c=cl[i]
    model=model_pip[i][0]
    y_pred_proba=c.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:,1])
    auc=roc_auc_score(y_test,y_pred_proba[:,1])
    #sub_fig_title=fig.add_subplot(2,5,i+1).set_title(model)
    plot_map=sns.lineplot(x=fpr,y=tpr,label=model+': AUC='+str(round(auc,4)))
plot_map.set_xlabel('False Positive Rate')
plot_map.set_ylabel('True Positive Rate')
