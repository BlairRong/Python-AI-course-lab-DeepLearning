import contractions
import re
from string import punctuation

import nltk  #pip install nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def clean_text(text):
    #remove contractions 收缩
    text = contractions.fix(text)
    #make lowercase
    text = text.lower()
    #remove punctuation 标点符号
    text = re.sub('[%s]' % re.escape(punctuation), '', text)
    #remove numbers
    text = re.sub(r'\w*\d\w*', '', text) #https://regex101.com/ ?????
    #remove stop words
    stopwords = [stopword.strip() for stopword in open('./3.16_class_text/data/stopwords_en.txt','r')]
    return ' '.join([word for word in text.split() if word not in stopwords])

text = "I read this book for the first time in 1987, and it's still one of my favorites!"

#fixed = contractions.fix(text)
#print(fixed)

#print(text.split()) #output: ['I', 'read', 'this', 'book', 'for', 'the', 'first', 'time', 'in', '1987,', 'and', "it's", 'still', 'one', 'of', 'my', 'favorites!']
#for stopword in open('./3.16_class_text/data/stopwords_en.txt','r'): #'r'means read list/'w' means add value
    #print(stopword.strip())
    
cleaned_text = clean_text(text)
print(cleaned_text)

#read book time favorites #by using kaggle stopwords

#teacher: read book first time still one favorites #different stopwords