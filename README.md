# Fraud Email Detecting System

![Python 3.8.10](https://img.shields.io/badge/python-3.8.10-yellow.svg)

Software system for detecting malicious content in text data using the Doc2vec algorithm.

Architecture is desctdescribed in [conference paper](https://hsse.spbstu.ru/userfiles/files/Sovremennie-tehnologii-v-teorii-i-praktike-programmirovaniya-2022-s-titulami.pdf) p/
160-170.

## Usage
Web application was built using AWS EC2 and AWS Load Balancer. [Link](http://frauddetection-lb-1094356394.eu-north-1.elb.amazonaws.com/).

### Installation
- Automatic installation
> startup.sh

- Manual installation
```bash
# install the required packages
python3 -m pip install -r requirements.txt

# train model
python3 train.py

# test model
python3 evaluate.py

# launch local web app
streamlit run model_deployment.py
```

## Code
The programs have been written in the python language. The list below will help understanding the role of each coding file.
1. Dataset: the folder contains the dataset for the project. 
   - [Enron dataset](http://www.cs.cmu.edu/~./enron/)
   - [Phishing dataset](https://monkey.org/~jose/phishing/)
2. Notebook: Jupyter notebooks of this program for every dataset.
3. dumped_models: In this folder there are saved Doc2Vec model and ten classifcation models.
4. balanced_dataset.py, imbalanced_dataset: the entire program code for every type of dataset in a python file.
5. load_data.py: load the data from Dataset.
6. preprocess.py: functions for cleaning text before training the model.
7. train.py: the stage of training model with Doc2vec algorithm and other machine learning classification algorithms.
8. evaluate.py: evaluate the model with different methods, such as: accuracy score, confusion matrix, ROC curve, Area Under the Curve, Precision and Recall, F1 score.
9. data_X_test.npy, data_y_test.npy: save numpy data X_test and y_test for evaluating after word embeddings by Doc2Vec algorithm.
10. unit-test.py: unit test for program.
11. startup.sh: bash script for automatic installation program.
12. requirements.txt: the required packages of program. 

## Results
![image](https://user-images.githubusercontent.com/57873174/170865358-8d927c51-441d-4f7a-907e-7da322317839.png)

![image](https://user-images.githubusercontent.com/57873174/170865386-82445fdf-62e9-44e5-87bc-756043f76bbb.png)
