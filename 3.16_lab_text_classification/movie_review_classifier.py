#Part 1: Build the Classifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Dataset
# Training corpus
corpus = [
    "the movie was fantastic and i loved every part of it",
    "an absolute masterpiece with brilliant acting",
    "the film was boring and too long",
    "i really enjoyed the story and the visuals",
    "the plot was terrible and the acting was even worse",
    "what a wonderful experience, highly recommend",
    "not worth my time, very disappointing",
    "a truly great film, i will watch it again",
    "the script was weak and the characters were flat",
    "an amazing journey from start to finish"
]

# Corresponding labels, will be Y
categories = [
    "Positive", "Positive", "Negative", "Positive", "Negative",
    "Positive", "Negative", "Positive", "Negative", "Positive"
]

# Test corpus
test_corpus = [
    "the movie was great",
    "i hated the film",
    "a boring and bad story",
    "absolutely loved it"
]

# True labels for test set (for evaluation)
true_test_labels = ["Positive", "Negative", "Negative", "Positive"]



#Step1: Vectorize the Text 把文本转换成数值向量
vectorizer = CountVectorizer(ngram_range=(1,2))   # Start with unigrams only(1,1) then change to (1,2)
X_train = vectorizer.fit_transform(corpus)        # Learn vocabulary and transform training data #学习词汇 并把文字转成文字+向量数字
X_test = vectorizer.transform(test_corpus)        # Transform test data using same vocabulary 使用相同的词汇表转换测试数据

print("Vocabulary size:", len(vectorizer.vocabulary_)) 
#output: Vocabulary size: 55
#change ngram_range from (1,1) to (1,2) output: Vocabulary size: 117

print("Feature names:", vectorizer.get_feature_names_out())

"""
output:
Feature names: ['absolute' 'acting' 'again' 'amazing' 'an' 'and' 'boring' 'brilliant'
'characters' 'disappointing' 'enjoyed' 'even' 'every' 'experience'
'fantastic' 'film' 'finish' 'flat' 'from' 'great' 'highly' 'it' 'journey'
'long' 'loved' 'masterpiece' 'movie' 'my' 'not' 'of' 'part' 'plot'
'really' 'recommend' 'script' 'start' 'story' 'terrible' 'the' 'time'
'to' 'too' 'truly' 'very' 'visuals' 'was' 'watch' 'weak' 'were' 'what'
'will' 'with' 'wonderful' 'worse' 'worth']

change ngram_range from (1,1) to (1,2)
output:
Feature names: ['absolute' 'absolute masterpiece' 'acting' 'acting was' 'again' 'amazing'
'amazing journey' 'an' 'an absolute' 'an amazing' 'and' 'and loved'
'and the' 'and too' 'boring' 'boring and' 'brilliant' 'brilliant acting'
'characters' 'characters were' 'disappointing' 'enjoyed' 'enjoyed the'
'even' 'even worse' 'every' 'every part' 'experience' 'experience highly'
'fantastic' 'fantastic and' 'film' 'film was' 'film will' 'finish' 'flat'
'from' 'from start' 'great' 'great film' 'highly' 'highly recommend' 'it'
'it again' 'journey' 'journey from' 'long' 'loved' 'loved every'
'masterpiece' 'masterpiece with' 'movie' 'movie was' 'my' 'my time' 'not'
'not worth' 'of' 'of it' 'part' 'part of' 'plot' 'plot was' 'really'
'really enjoyed' 'recommend' 'script' 'script was' 'start' 'start to'
'story' 'story and' 'terrible' 'terrible and' 'the' 'the acting'
'the characters' 'the film' 'the movie' 'the plot' 'the script'
'the story' 'the visuals' 'time' 'time very' 'to' 'to finish' 'too'
'too long' 'truly' 'truly great' 'very' 'very disappointing' 'visuals'
'was' 'was boring' 'was even' 'was fantastic' 'was terrible' 'was weak'
'watch' 'watch it' 'weak' 'weak and' 'were' 'were flat' 'what'
'what wonderful' 'will' 'will watch' 'with' 'with brilliant' 'wonderful'
'wonderful experience' 'worse' 'worth' 'worth my']
"""


#Step2: Train a Classifier (SVM) 训练数据的训练分类器
model = SVC(kernel='linear')     #Support Vector Classification
model.fit(X_train, categories)


#Step3: Predict and Evaluate 测试句子的情感倾向
predictions = model.predict(X_test)
print("Predictions:", predictions)
#output: Predictions: ['Positive' 'Positive' 'Positive' 'Positive']
#Two wrong, Two correct

accuracy = accuracy_score(true_test_labels, predictions)
print(f"Test Accuracy: {accuracy*100:.1f}%")
#Test Accuracy: 50.0%






#Part 2: Investigate the Model
# Q1. What happends if you change ngram_range from (1,1) to (1,2)?
"""
Effect:
The vocabulary now includes both single words (unigrams) and two-word phrases (bigrams).
The feature matrix becomes larger because more combinations are considered.
The model may capture some local word order, e.g., "not good" as a phrase.
"""

# Q2: Why might bigrams help in sentiment analysis? 为什么二元语法（2歌词连起来）有助于情感分析
""" 
Bigrams can capture negations (e.g., "not good") and common sentiment-bearing phrases (e.g., "absolutely loved") 
that are more than the sum of individual words. 
In bag-of-words, "not" and "good" are treated independently, which loses the negation meaning. 
Bigrams preserve that combination.
"""

# Q3:Which test sentences were classified incorrectly?
""" 
"i hated the film",
"a boring and bad story",
"""

# Q4: 4. Why might the sentence "the movie was not good" be difficult for a simple bag-of-words model?
""" 
A bag-of-words model ignores word order. 
The words "not" and "good" appear separately, 
but the model does not know they are connected. 
It sees the presence of "good" (a positive word) and might predict "Positive" 
even though the actual sentiment is negative due to the negation.
"""

# Q5: What would likely happen if you had 1000 reviews instead of 10?
""" 
With more data:

The model would generalize better and be more robust.
Accuracy on real-world test sets would likely improve.
Overfitting would be reduced because more examples cover more linguistic variations.
The vocabulary would be larger, capturing more diverse expressions.
"""


#Part 3 in the improved_model.py







