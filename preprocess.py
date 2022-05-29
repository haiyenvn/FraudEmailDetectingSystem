from bs4 import BeautifulSoup
import re
from textblob import TextBlob
import nltk
from nltk.corpus import wordnet 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def cleanText(text):
    text = BeautifulSoup(text, "html.parser").text
    # remove all special characters
    text = re.sub('[^A-Za-z0-9 ]+',r'', text)
    text = re.sub(r'\|\|\|', r' ', text) 
    #remove http(s) links with <URL>
    text = re.sub(r'http\S+', r'<URL>', text)
    #remove unwanted lines starting from special charcters
    text = re.sub(r'\n: \'\'.*','', text)
    text = re.sub(r'\n!.*','', text)
    text = re.sub(r'^:\'\'.*','', text)
    #remove non-breaking new line characters
    text = re.sub(r'\n',' ',text)
    text = text.lower()
    text = text.replace('x', '')
    return text

def spelling_correction(text):
    text = TextBlob(text)
    text = text.correct()
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = nltk.word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return ' '.join(filtered_sentence)
    
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens
