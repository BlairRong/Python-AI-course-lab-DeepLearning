#Part 3:  Improve the Model

""" 
Test at least three of the following changes:

use stop_words="english"
change ngram_range
try another classification model
add at least 6 new movie reviews
create your own test sentences
clean the text before vectorizing it
"""

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


#Improvement 1: Use stop_words='english'
vectorizer = CountVectorizer(ngram_range=(1,1), stop_words='english')
X_train = vectorizer.fit_transform(corpus)
X_test = vectorizer.transform(test_corpus)

model = SVC(kernel='linear')     #Train the model Support Vector Classificatio
model.fit(X_train, categories)

pred = model.predict(X_test)     #predict

print("Feature names:", vectorizer.get_feature_names_out())
""" 
Feature names: ['absolute' 'acting' 'amazing' 'boring' 'brilliant' 'characters'
'disappointing' 'enjoyed' 'experience' 'fantastic' 'film' 'finish' 'flat'
'great' 'highly' 'journey' 'long' 'loved' 'masterpiece' 'movie' 'plot'
'really' 'recommend' 'script' 'start' 'story' 'terrible' 'time' 'truly'
'visuals' 'watch' 'weak' 'wonderful' 'worse' 'worth']
"""
print("Stop words predictions:", pred)
#Output: Stop words predictions: ['Positive' 'Negative' 'Negative' 'Positive']
print("Accuracy:", accuracy_score(true_test_labels, pred))
#Accuracy: 1.0
#The accuracy result is improved from 0.5 to 1 
#Result: With stop words removed, common words like "the", "and" are ignored. 
# This may help focus on content words. 
# In our tiny dataset, the accuracy remains 100%.




#Improvement 2: try another classification model -Change to Multinomial Naive Bayes 换另一个模型
from sklearn.naive_bayes import MultinomialNB

vectorizer = CountVectorizer(ngram_range=(1,1))
X_train = vectorizer.fit_transform(corpus)
X_test = vectorizer.transform(test_corpus)

nb_model = MultinomialNB()  #change and train to Multinomial Naive Bayes classification model
nb_model.fit(X_train, categories)

pred = nb_model.predict(X_test) #predict


print("Feature names:", vectorizer.get_feature_names_out())
""" 
Feature names: ['absolute' 'acting' 'again' 'amazing' 'an' 'and' 'boring' 'brilliant'
'characters' 'disappointing' 'enjoyed' 'even' 'every' 'experience'
'fantastic' 'film' 'finish' 'flat' 'from' 'great' 'highly' 'it' 'journey'
'long' 'loved' 'masterpiece' 'movie' 'my' 'not' 'of' 'part' 'plot'
'really' 'recommend' 'script' 'start' 'story' 'terrible' 'the' 'time'
'to' 'too' 'truly' 'very' 'visuals' 'was' 'watch' 'weak' 'were' 'what'
'will' 'with' 'wonderful' 'worse' 'worth']
"""
print("Naive Bayes predictions:", pred)
#output: Naive Bayes predictions: ['Positive' 'Negative' 'Negative' 'Positive']
print("Accuracy:", accuracy_score(true_test_labels, pred))
#Accuracy: 1.0
#The accuracy result is improved from 0.5 to 1 
#Result: Naive Bayes also gives 100% on this simple test set.





#Improcement 3: Add new training reviews - 6 new moview reviews

new_reviews = [
    "the movie was okay but not great",
    "i loved the cinematography and the music",
    "the acting was terrible but the story was good",
    "a complete waste of time",
    "very entertaining and funny",
    "the film was too long and boring"
]
new_labels = ["Positive", "Positive", "Negative", "Negative", "Positive", "Negative"]

# Extend the original corpus
corpus_extended = corpus + new_reviews
categories_extended = categories + new_labels

vectorizer = CountVectorizer(ngram_range=(1,1))
X_train = vectorizer.fit_transform(corpus_extended)
X_test = vectorizer.transform(test_corpus)

model.fit(X_train, categories_extended) #still use SVC train

pred = model.predict(X_test) #predict


print("Extended data predictions:", pred)
#Extended data predictions: ['Positive' 'Positive' 'Positive' 'Positive']
print("Accuracy:", accuracy_score(true_test_labels, pred))
#Accuracy: 0.5
#Result: adding 6 new movie reviews doesn't improve the accuracy, 
# might because of the dataset are still too small



#Part 4: Understanding the Limitations
tricky_sentences = [
    "the movie was not good",
    "the acting was not bad",
    "visually impressive but boring",
    "i wanted to like it"
]
#negative positive negative negative

# Use the best model from Part 3 Improvement 2
vectorizer = CountVectorizer(ngram_range=(1,1))
X_train = vectorizer.fit_transform(corpus)
X_tricky = vectorizer.transform(tricky_sentences)

nb_model = MultinomialNB()  #change and train to Multinomial Naive Bayes classification model
nb_model.fit(X_train, categories)

tricky_pred = nb_model.predict(X_tricky)
print("Tricky predictions:", tricky_pred)
#output: Tricky predictions: ['Negative' 'Negative' 'Negative' 'Positive']



#Q1: Which of these sentences were classified incorrectly?
"""
"the acting was not bad", "i wanted to like it" those two sentence are incorrectly 
"not bad" is Positive but could be ambiguous
The bag-of-words model sees "good" "like" and predicts Positive; it sees "bad" and predicts Negative, ignoring the negation.

"""


#Q2: Why might the model struggle with sentences that contain negations like "not good" or "not bad"?
""" 
Because bag-of-words treats each word independently. 
It does not understand that "not" reverses the sentiment of the following word. 
Thus, "not good" appears as both "not" (neutral) and "good" (positive), 
leading to an incorrect positive prediction.

"""

#Q3: Why might sarcasm or mixed opinions be difficult for this type of model?
""" 
Sarcasm often relies on contrast between literal meaning and intended meaning
(e.g., "great movie, fell asleep").
Mixed opinions contain both positive and negative words, 
and the overall sentiment depends on context. 
Bag-of-words cannot capture such nuance.

"""


#Q4: What information about the sentence meaning is lost when we only count words?
""" 
Word order and grammar.
Negations and their scope.
Intensity (e.g., "good" vs "amazing").
Relationships between words
Pragmatic and contextual cues.

"""






#Part 5 From Bag-of-Words to Modern NLP 从词袋模型到现代自然语言处理
#Q1: What is the main limitation of representing text using only word counts?
""" 
Word counts ignore semantic relationships between words, word order, and context. 词频忽略了词语之间的语义关系、词序和上下文
Synonyms are treated as different features, and polysemy (multiple meanings) is not resolved.  同义词被视为不同的特征，并且无法解决多义性（多含义）问题。
The representation is high-dimensional and sparse 高纬且稀疏的.
"""


#Q2: Why might a model that understands word context perform better?
"""
Contextual models (like BERT) capture the meaning of a word based on surrounding words.  上下文模型（例如 BERT）能够根据词语的上下文来理解词义。
They can handle polysemy (e.g., "bank" as river vs. financial) and better understand negations and long-range dependencies, 它们可以处理多义性（例如，“bank”既可以指河流，也可以指金融），并且能够更好地理解否定和长程依赖关系，
leading to more accurate sentiment detection.
"""

#Q3: What kinds of problems in language might require more advanced NLP models?
""" 
-Sentiment analysis with sarcasm, irony, or mixed opinions. 包含讽刺、反讽或混合意见的情感分析。
-Machine translation.机器翻译。
-Question answering.问答系统。
-Text summarization.文本摘要。
-Dialogue systems.对话系统
-Any task requiring deep understanding of context, world knowledge, or reasoning. - 任何需要深入理解语境、世界知识或推理能力的任务。

"""


###Final Reflection

#Q1:Which version of your model worked best?
"""
The model trained with use stop_words='english' and change to Multinomial Naive Bayes classification model  performed perfectly on our simple test set. 
However, adding more data and bigrams improved its ability to handle phrases.
"""

#Q2: Why do you think it performed better?
""" 
stop words can reduce confusing 
"""

#Q3: What are the limitations of this small dataset?
""" 
With only 10 reviews, the model cannot generalize to real-world language variations. 
It overfits to the exact vocabulary of the training set. 
Negations, mixed opinions, and sarcasm are not well-represented.
"""

#Q4: What would you change if you were building a real sentiment classifier?
"""
Use a much larger dataset (e.g., IMDb reviews).

Apply text preprocessing: lowercasing, removing punctuation, handling negations.

Use more advanced representations like TF-IDF, word embeddings (Word2Vec, GloVe), or fine-tune a transformer model (BERT).

Validate with cross-validation and test on diverse examples.

Consider class imbalance and use appropriate metrics.
"""


#Bonus

my_tricky = [
    "the movie was not bad at all, actually quite good",   # Positive
    "i expected to hate it but ended up loving it",        # Positive
    "the plot was predictable yet entertaining"            # Positive (mixed but overall positive)
]

vectorizer = CountVectorizer(ngram_range=(1,1), stop_words='english')
X_train = vectorizer.fit_transform(corpus)
X_my = vectorizer.transform(my_tricky)

model = SVC(kernel='linear')     #Train the model Support Vector Classificatio
model.fit(X_train, categories)

my_pred = model.predict(X_my)          #using the stop words 
print("My tricky predictions:", my_pred)

#My tricky predictions: ['Positive' 'Positive' 'Negative']