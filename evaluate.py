from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd 
import pickle
import os
from numpy import load

X_test = load("data_X_test.npy")
y_test = load("data_y_test.npy")

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

acc=[]
f1=[]
precision=[]
recall=[]
roc_auc=[]
conf_mat=[]
cl=[]
model_name=[]

for filename in os.listdir("dumped_models/classification"):
    if filename.endswith(".pkl"):
        with open('dumped_models/classification/{}'.format(filename),'rb') as f:
            model=pickle.load(f)
            y_pred=model.predict(X_test)
            acc.append(accuracy_score(y_test,y_pred))
            f1.append(f1_score(y_test,y_pred))
            precision.append(precision_score(y_test,y_pred))
            recall.append(recall_score(y_test,y_pred))
            roc_auc.append(roc_auc_score(y_test,y_pred))
            conf_mat.append(confusion_matrix(y_test,y_pred))
            cl.append(model)
            model_name.append(filename)
            
result=pd.DataFrame({'Model': model_name, 'Accuracy': acc, 'F1-score': f1, 'Precision': precision, 'Recall': recall})
print(result)

plt.rcParams.update({'font.size': 22})

fig=plt.figure(figsize=(20,10))

for i in range(0,len(cl)):
    c=cl[i]
    model=model_name[i][0]
    y_pred_proba=c.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:,1])
    auc=roc_auc_score(y_test,y_pred_proba[:,1])
    #sub_fig_title=fig.add_subplot(2,5,i+1).set_title(model)
    plot_map=sns.lineplot(x=fpr,y=tpr,label=model+': AUC='+str(round(auc,4)))
plot_map.set_xlabel('False Positive Rate')
plot_map.set_ylabel('True Positive Rate')
fig.savefig("ROCs")

fig=plt.figure(figsize=(20,50))

for i in range(0,len(conf_mat)):
    cm_con=conf_mat[i]
    model=model_name[i][0]
    sub_fig_title=fig.add_subplot(5,2,i+1).set_title(model)
    plot_map=sns.heatmap(cm_con,annot=True,cmap='Greens_r',fmt='g')
    plot_map.set_xlabel('Predicted_Values')
    plot_map.set_ylabel('Actual_Values')
fig.savefig("confution-matrix")