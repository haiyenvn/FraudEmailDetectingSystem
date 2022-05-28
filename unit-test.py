from turtle import shape
import tkinter as TK
import unittest
import doctest
import os
from matplotlib.pyplot import text
import pandas as pd
import load_data
import preprocess
#import train2

data=pd.DataFrame()

class TestPreprocessingMethods(unittest.TestCase):
    def test_load_data(self):
        self.assertEqual(load_data.load_data_phishing(data).shape, (4158,2))

    def test_clean_text(self):
        text = 'He.ll-o worl/d!'
        text_expected = 'hello world'
        self.assertEqual(preprocess.cleanText(text), text_expected)

    def test_spelling_correction(self):
        text = 'I am abl to judg, afters long attnding to the sbject creditcard, the condiions of lfe apear to act in two waysdirectly on the whle organsaton.'
        text_expected = 'I am all to judge, after long attending to the subject creditcard, the conditions of life appear to act in two waysdirectly on the while organisation.'
        self.assertEqual(preprocess.spelling_correction(text), text_expected)

    def test_remove_stopwords(self):
        text = 'I am all to judge, after long attending to the subject creditcard, the conditions of life appear to act in two waysdirectly on the while organisation.'
        text_expected = 'I judge , long attending subject creditcard , conditions life appear act two waysdirectly organisation .'
        self.assertEqual(preprocess.remove_stopwords(text), text_expected)

    def test_tokenize_text(self):
        text = 'Hello, world!'
        self.assertEqual(preprocess.tokenize_text(text), ['hello', 'world'])

if __name__ == '__main__':
    unittest.main()