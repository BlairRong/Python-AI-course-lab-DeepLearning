"""
Advanced Track: Classical Model Experiments
Compare CountVectorizer + classifier under different settings and against Hugging Face.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from transformers import pipeline

# ------------------------------
# 1. Data Preparation
# ------------------------------
# Original training data (10 reviews)
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
categories = ["Positive", "Positive", "Negative", "Positive", "Negative",
            "Positive", "Negative", "Positive", "Negative", "Positive"]

# try Additional reviews to see if more data helps
extra_reviews = [
    "i absolutely loved this movie, it was amazing",
    "the acting was terrible and the story made no sense",
    "a decent film, not great but not bad either",
    "fantastic visuals but the plot was lacking",
    "one of the worst movies i have ever seen"
]
extra_categories = ["Positive", "Negative", "Positive", "Positive", "Negative"]

# Extended dataset
corpus_extended = corpus + extra_reviews
categories_extended = categories + extra_categories

# Test sentences (same as before)
test_sentences = [
    "the movie was great",
    "i hated the film",
    "the movie was not good",
    "the acting was not bad",
    "visually impressive but boring",
    "i wanted to like it"
]
true_labels = ["Positive", "Negative", "Negative", "Positive", "Negative", "Negative"]


# ------------------------------
# 2. Hugging Face baseline (run once)
# ------------------------------
print("HUGGING FACE BASELINE")

sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
hf_results = sentiment_pipe(test_sentences)
hf_preds = [res['label'] for res in hf_results]  # e.g., 'POSITIVE', 'NEGATIVE'
# Convert to match true_labels format
hf_preds_converted = ['Positive' if p == 'POSITIVE' else 'Negative' for p in hf_preds]
hf_acc = accuracy_score(true_labels, hf_preds_converted)
print(f"Hugging Face Accuracy: {hf_acc*100:.1f}%")
print("Raw predictions:", hf_preds)
print("Converted predictions:", hf_preds_converted)
print()


"""
Output:
Hugging Face Accuracy: 66.7%
Raw predictions: ['POSITIVE', 'NEGATIVE', 'NEGATIVE', 'POSITIVE', 'POSITIVE', 'POSITIVE']
Converted predictions: ['Positive', 'Negative', 'Negative', 'Positive', 'Positive', 'Positive']
"""

# ------------------------------
# 3. Function to evaluate classical model
# ------------------------------
def evaluate_classical(corpus, categories, test_sentences, true_labels,
                    lowercase=True, remove_stopwords=False, ngram_range=(1,1), 
                    classifier='SVC', description=""):
    """
    Train and evaluate a CountVectorizer + classifier model.
    """
    print(description)

    # try Text preprocessing options 文本清洗
    if lowercase and remove_stopwords:
        # Using built-in stopwords from CountVectorizer
        vectorizer = CountVectorizer(lowercase=lowercase, stop_words='english', ngram_range=ngram_range)
    elif lowercase:
        vectorizer = CountVectorizer(lowercase=lowercase, ngram_range=ngram_range)
    else:
        vectorizer = CountVectorizer(lowercase=False, ngram_range=ngram_range)

    X_train = vectorizer.fit_transform(corpus)
    X_test = vectorizer.transform(test_sentences)
    
    # try different modles(SVC vs Naive Bayes)
    if classifier == 'SVC':
        clf = SVC(kernel='linear')
    elif classifier == 'NB':
        clf = MultinomialNB()
    else:
        raise ValueError("Unknown classifier")

    clf.fit(X_train, categories)
    preds = clf.predict(X_test)
    acc = accuracy_score(true_labels, preds)

    print(f"Classifier: {classifier}")
    print(f"Lowercase: {lowercase}, Stopwords removed: {remove_stopwords}, ngram_range: {ngram_range}")
    print(f"Predictions: {list(preds)}")
    print(f"Accuracy: {acc*100:.1f}%")
    print()
    return preds, acc


# ------------------------------
# 4. Run experiments 运行实验
# ------------------------------

print("CLASSICAL MODEL EXPERIMENTS")

# Baseline (original data, no cleaning, unigrams, SVC)
evaluate_classical(corpus, categories, test_sentences, true_labels,
                lowercase=True, remove_stopwords=False, ngram_range=(1,1),
                classifier='SVC', description="Baseline (original data, unigrams, SVC)")
""" 
Output:
Baseline (original data, unigrams, SVC)
Classifier: SVC
Lowercase: True, Stopwords removed: False, ngram_range: (1, 1)
Predictions: ['Positive', 'Positive', 'Negative', 'Negative', 'Positive', 'Positive']
Accuracy: 33.3%
"""


# 4.1 With text cleaning (lowercase + stopwords) 🌟
evaluate_classical(corpus, categories, test_sentences, true_labels,
                lowercase=True, remove_stopwords=True, ngram_range=(1,1),
                classifier='SVC', description="With stopwords removed")
""" 
Output:
With stopwords removed
Classifier: SVC
Lowercase: True, Stopwords removed: True, ngram_range: (1, 1)
Predictions: ['Positive', 'Negative', 'Positive', 'Positive', 'Negative', 'Positive']
Accuracy: 66.7%
"""


# 4.2 Different ngram_range
evaluate_classical(corpus, categories, test_sentences, true_labels,
                lowercase=True, remove_stopwords=False, ngram_range=(1,2),
                classifier='SVC', description="ngram_range=(1,2)")
""" 
Output:
ngram_range=(1,2)
Classifier: SVC
Lowercase: True, Stopwords removed: False, ngram_range: (1, 2)
Predictions: ['Positive', 'Positive', 'Positive', 'Negative', 'Positive', 'Positive']
Accuracy: 16.7%

"""

evaluate_classical(corpus, categories, test_sentences, true_labels,
                lowercase=True, remove_stopwords=False, ngram_range=(2,2),
                classifier='SVC', description="ngram_range=(2,2)")
""" 
Output:
Classifier: SVC
Lowercase: True, Stopwords removed: False, ngram_range: (2, 2)
Predictions: ['Positive', 'Positive', 'Positive', 'Positive', 'Positive', 'Positive']
Accuracy: 33.3%

"""


# 4.3 Different classifier (Naive Bayes) 🌟
evaluate_classical(corpus, categories, test_sentences, true_labels,
                lowercase=True, remove_stopwords=False, ngram_range=(1,1),
                classifier='NB', description="Naive Bayes (unigrams)")
""" 
Output:
Naive Bayes (unigrams)
Classifier: NB
Lowercase: True, Stopwords removed: False, ngram_range: (1, 1)
Predictions: ['Positive', 'Negative', 'Negative', 'Negative', 'Negative', 'Positive']
Accuracy: 66.7%

"""


# 4.4 With additional reviews
evaluate_classical(corpus_extended, categories_extended, test_sentences, true_labels,
                lowercase=True, remove_stopwords=False, ngram_range=(1,1),
                classifier='SVC', description="With 5 extra reviews (unigrams, SVC)")
""" 
Output:
With 5 extra reviews (unigrams, SVC)
Classifier: SVC
Lowercase: True, Stopwords removed: False, ngram_range: (1, 1)
Predictions: ['Positive', 'Positive', 'Positive', 'Negative', 'Positive', 'Positive']
Accuracy: 16.7%

"""

# 4.5 try combination: ngram=(1,2) + extra data + NB maybe
evaluate_classical(corpus_extended, categories_extended, test_sentences, true_labels,
                lowercase=True, remove_stopwords=False, ngram_range=(1,2),
                classifier='NB', description="Extra data + ngram=(1,2) + Naive Bayes")
""" 
Output:
Extra data + ngram=(1,2) + Naive Bayes
Classifier: NB
Lowercase: True, Stopwords removed: False, ngram_range: (1, 2)
Predictions: ['Positive', 'Negative', 'Positive', 'Negative', 'Positive', 'Positive']
Accuracy: 33.3%

"""

# 4.6 try with text cleaning + Naive Bayes
evaluate_classical(corpus, categories, test_sentences, true_labels,
                lowercase=True, remove_stopwords=True, ngram_range=(1,1),
                classifier='NB', description="Text cleaning + Naive Bayes")
""" 
Output:
Text cleaning + Naive Bayes
Classifier: NB
Lowercase: True, Stopwords removed: True, ngram_range: (1, 1)
Predictions: ['Positive', 'Positive', 'Positive', 'Positive', 'Negative', 'Positive']
Accuracy: 50.0%

"""



# ------------------------------
# 5. Extra Analysis 
# ------------------------------

#How do the result change?
"""
Based on the experiments:

- The Hugging Face model achieved 🌟66.7% accuracy (correct on 4 out of 6 sentences). 
It handled negation well ('not good' → Negative, 'not bad' → Positive) 
but failed on mixed sentiment and the implied negative sentence.

- The classical model's performance varied significantly with preprocessing choices:
  * Baseline (SVC, unigrams): 33.3% - only got one correct.
  * Removing stopwords with SVC boosted accuracy to 🌟66.7% (tied with Hugging Face) because it focused on content words, though it still made mistakes on some sentences.
  * Using bigrams (1,2) with SVC dropped accuracy to 16.7%, indicating that adding bigrams introduced noise without enough data.
  * Bigrams only (2,2) with SVC gave 33.3%, same as baseline.
  * Naive Bayes with unigrams achieved 🌟66.7%, matching the best classical result.
  * Adding extra reviews did not help the SVC model (accuracy dropped to 16.7%), likely because the new reviews introduced conflicting patterns.
  * Extra data with (1,2) bigrams and Naive Bayes gave 33.3%.
  * Try the combination (text cleaning + Naive Bayes) improve to 50%.
"""


#When does the classical model improve?
"""
- With proper preprocessing (stopwords removal) and a suitable classifier (Naive Bayes or SVC), it can reach 66.7% on this test set, matching Hugging Face.
- It performs well on simple sentences without negation or complex structure.它在不包含否定词或复杂结构的简单句上表现良好。
- Naive Bayes sometimes outperforms SVC on this small dataset due to its probabilistic nature.由于朴素贝叶斯算法的概率特性，它有时在这个小型数据集上的表现优于 SVC 模型。
"""


#Are there cases where it performs as well as or better than Hugging Face?
"""
- Yes, the stopwords-removed SVC and the unigram Naive Bayes both achieved 66.7%, the same as Hugging Face.
- However, Hugging Face demonstrates better understanding of negation and mixed sentiment, as seen in the predictions. 然而,正如预测结果所示,Hugging Face模型在理解否定和混合情感方面表现更佳。
The classical model's success on 'not good' and 'not bad' is inconsistent - it sometimes gets them right by chance based on word presence.经典模型在“不好”和“不坏”这两个词上的准确率并不稳定——有时它只是基于词语出现情况的偶然性而做出正确判断。

The classical model's main limitation remains its inability to understand word order and context, 经典模型的主要局限在于它无法理解词序和上下文,而这对于处理否定和对比至关重要。即使准确率与Hugging Face模型相当,其底层推理的稳健性也较差。
which is essential for handling negation and contrast. 
Even when accuracy matches Hugging Face, the underlying reasoning is less robust.
"""



