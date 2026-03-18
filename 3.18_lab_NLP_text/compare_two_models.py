"""
Lab: Machine Learning - Comparing NLP Approaches - Classical vs. Hugging Face

Goal:
In this task, you will compare two different ways of working with text:
A classical machine learning model (CountVectorizer + classifier)
A modern AI model using Hugging Face
The goal is to understand the differences between these approaches.
"""


# ------------------------------
# Setup and Data
# ------------------------------
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from transformers import pipeline
import numpy as np

# Training data from previous 3.16 lab 
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

# Test sentences
test_sentences = [
    "the movie was great",
    "i hated the film",
    "the movie was not good",
    "the acting was not bad",
    "visually impressive but boring",
    "i wanted to like it"
]


# True labels for test set (for evaluation)
true_test_labels = ["Positive", "Negative", "Negative", "Positive", "Negative", "Negative"]


# ------------------------------
# Part 1 – My Previous Model 3.16lab (CountVectorizer + Classifiel:SVC)
# ------------------------------

print("PART 1: CountVectorizer + SVC Model")

# Vectorize the training data
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(corpus)

# Train SVC classifier
clf = SVC(kernel='linear')
clf.fit(X_train, categories)

# Predict on test sentences
X_test = vectorizer.transform(test_sentences)
pred_classical = clf.predict(X_test)

print("\nPredictions on test sentences:")
for sent, pred in zip(test_sentences, pred_classical):
    print(f"  '{sent}' -> {pred}")

accuracy = accuracy_score(true_test_labels, pred_classical)
print(f"Test Accuracy: {accuracy*100:.1f}%")
"""
output:
PART 1: CountVectorizer + SVC Model

Predictions on test sentences:
'the movie was great' -> Positive
'i hated the film' -> Positive x
'the movie was not good' -> Negative
'the acting was not bad' -> Negative x
'visually impressive but boring' -> Positive x
'i wanted to like it' -> Positive  x
Test Accuracy: 33.3%
"""




# ------------------------------
# Part 2 – Hugging Face Model Pipeline
# ------------------------------

print("PART 2: Hugging Face Sentiment Pipeline")

# Load the default sentiment pipeline (distilbert-base-uncased-finetuned-sst-2-english)
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Run on test sentences
hf_results = sentiment_pipe(test_sentences)

print("\nPredictions on test sentences:")
for sent, res in zip(test_sentences, hf_results):
    print(f"  '{sent}' -> {res['label']} (confidence: {res['score']:.4f})")


"""
Output:
PART 2: Hugging Face Sentiment Pipeline
Device set to use cpu

Predictions on test sentences:
'the movie was great' -> POSITIVE (confidence: 0.9999)
'i hated the film' -> NEGATIVE (confidence: 0.9997)
'the movie was not good' -> NEGATIVE (confidence: 0.9998)
'the acting was not bad' -> POSITIVE (confidence: 0.9987)
'visually impressive but boring' -> POSITIVE (confidence: 0.6628) x
'i wanted to like it' -> POSITIVE (confidence: 0.9997) x
"""


# ------------------------------
# Part 3 – Compare Results
# ------------------------------
print("PART 3: Comparison")

print("\n{:<35} {:<15} {:<15}".format("Sentence", "Classical", "HuggingFace"))

for sent, pred_c, res_h in zip(test_sentences, pred_classical, hf_results):
    print("{:<35} {:<15} {:<15}".format(sent, pred_c, res_h['label']))

""" 
Output:
PART 3: Comparison

Sentence                            Classical       HuggingFace    
the movie was great                 Positive        POSITIVE       
i hated the film                    Positive  X     NEGATIVE       
the movie was not good              Negative        NEGATIVE       
the acting was not bad              Negative  X     POSITIVE       
visually impressive but boring      Positive  X     POSITIVE   X    
i wanted to like it                 Positive  X     POSITIVE   X
"""

"""
Hugging Face correctly handled the two negation sentences (not good → Negative, not bad → Positive), while the classical model failed on both.

Hugging Face also correctly classified i hated the film as Negative, whereas the classical model missed it.

Both models struggled with mixed sentiment (visually impressive but boring) - classical got it wrong (Positive instead of Negative), Hugging Face also got it wrong (Positive), though with lower confidence (0.66).

Hugging Face incorrectly predicted i wanted to like it as Positive, which is also a subtle case (implied negative sentiment). The classical model also failed here.

Overall, Hugging Face performed better on negation and explicit sentiment, but both models have limitations on more complex sentences.

"""



# ------------------------------
# Part 4 – Analysis (answers)
# ------------------------------

#Q1 Which model performed better overall?
""" 
Hugging Face performed better because it correctly classified 4 out of 6 sentences (accuracy 66.7% vs. 33.3% for the classical model). 
It understood negations and direct sentiment more reliably.
"""

#Q2 Which sentences were difficult for your first model?
""" 
Sentences with negation (not bad), 
explicit negative (i hated the film), 
mixed sentiment (visually impressive but boring), 
and implied negative (i wanted to like it) were all difficult 

The classical model got two correct:
(the movie was great) It predicted Positive correctly for that one, but that might be because it saw "great" and predicted Positive.
(the movie was not good) It predicted Negative for "not good" - that's actually correct, but it may have been a lucky guess because it saw "not"? The classical model's bag-of-words often fails on negation, but here it correctly classified "not good" as Negative. 
However it failed on "not bad". So it's inconsistent.
"""

#Q3 Why is "the movie was not good" difficult?
""" 
In a bag-of-words model, the words "good" (positive) and "not" (neutral) are treated independently. 
The model may be biased by the presence of "good" and predict Positive, 
or it might learn that "not" often precedes negative words, 
but it cannot capture the composition. 
In my case, it actually predicted Negative correctly, so it might have seen "not" as a negative marker, 
but that's not guaranteed.
"""

#Q4 Why is "the acting was not bad" difficult?
""" 
Similarly, "bad" is a strong negative word; the model sees "bad" and may predict Negative, ignoring the negation. 
My classical model did exactly that: predicted Negative instead of Positive.
"""

#Q5 Which model seems to better understand: negation? mixed sentiment?
""" 
Hugging Face clearly understands negation (both "not good" → Negative, "not bad" → Positive). 
Mixed sentiment still challenges both models, but Hugging Face's confidence was lower for that sentence (0.66), indicating uncertainty.
"""

#Q6 Why does your first model struggle with meaning?
""" 
It ignores word order and treats each word independently. 
It cannot capture relationships like negation, contrast, or sarcasm.
"""

#Q7 Why does the Hugging Face model perform better?
""" 
It is a transformer pre-trained on large corpora and fine-tuned for sentiment. 这是一个经过大型语料库预训练并针对情感分析进行微调的Transformer模型。
It uses attention to understand context and word relationships, enabling it to interpret negation and subtle cues.它利用注意力机制来理解上下文和词语关系，从而能够解读否定词和细微的情感线索。
"""



# ------------------------------
# Part 5 – Reflectionn
# ------------------------------

#Q1 What is the main difference between: CountVectorizer + classifier & Hugging Face models
""" 
The main difference between the classical CountVectorizer+SVC approach and Hugging Face models is that the former treats text as a bag of independent words, losing word order and context, 前者将文本视为独立词袋，丢失了词序和上下文信息
while the latter uses deep learning to capture contextual relationships. 而后者则使用深度学习来捕捉上下文关系。
The classical model is fast, interpretable, and suitable for simple tasks with limited data, 经典模型速度快、易于解释，适用于数据量有限的简单任务，
but fails on negation and nuanced sentiment. 但在处理否定词和细微的情感表达方面表现不佳。
Hugging Face models require more resources but deliver superior accuracy on complex language understanding. Hugging Face 模型需要更多资源，但在理解复杂的语言方面具有更高的准确率。
"""

#Q2 When would you use each approach?
""" 
I would choose the classical approach for prototyping or when computational resources are limited, 我会选择经典方法用于原型设计或计算资源有限的情况，
Hugging Face for production systems where accuracy on subtle language phenomena is critical. 而对于生产系统，如果对细微的语言现象的准确率要求很高，我会选择 Hugging Face。
"""

# ------------------------------
# Bonus 🌟 – Try my own sentences
# ------------------------------
print("BONUS: Custom Sentences")

custom_sentences = [
    "Oh great, another Monday - just what I needed.",          # sarcasm - Negative
    "The movie started well but ended terribly.",              # mixed sentiment - Negative
    "I absolutely loved the first half, but the second half was a complete disaster."  # long text, mixed - Negative
]

print("\nCustom sentences:")
for s in custom_sentences:
    print(f"  '{s}'")

# Classical model predictions
X_custom = vectorizer.transform(custom_sentences)
pred_custom_classical = clf.predict(X_custom)

# Hugging Face predictions
hf_custom = sentiment_pipe(custom_sentences)

print("\n{:<60} {:<15} {:<15}".format("Sentence", "Classical", "HuggingFace"))

for sent, pred_c, res_h in zip(custom_sentences, pred_custom_classical, hf_custom):
    print("{:<60} {:<15} {:<15}".format(sent, pred_c, res_h['label']))


""" 
Output:
Sentence                                                                       Classical       HuggingFace    
Oh great, another Monday - just what I needed.                                  Positive  X      POSITIVE  x       
The movie started well but ended terribly.                                      Positive  X      NEGATIVE       
I absolutely loved the first half, but the second half was a complete disaster. Negative         NEGATIVE  
"""

#Q1 Do the models agree? 
""" 
The models agree on two of the three sentences: the first (sarcastic) and the third (mixed but dominantly negative). 
They disagree on the second sentence, where the classical model predicts Positive while Hugging Face correctly predicts Negative.
"""

#Q2 Why or why not?
""" 
The agreement on the first sentence occurs because both models rely on surface-level words like "great" and "needed", 
which are strongly associated with positive sentiment. The classical bag-of-words model cannot detect sarcasm, 
so it treats the sentence as literally positive. 
Hugging Face also fails to catch the sarcasm here - likely because the phrase is short and the positive words outweigh subtle cues.

The third sentence contains a clear negative conclusion ("complete disaster") that dominates the overall sentiment. 
Both models pick up on that strong negative signal, leading to agreement.

The disagreement on the second sentence highlights the key difference between the two approaches. 
The classical model only sees individual words: "started well" (positive) and "ended terribly" (negative). 
Because it ignores word order and the contrastive structure ("but"), it incorrectly focuses on the early positive word "well" and predicts Positive. 
Hugging Face, as a transformer, understands the sentence structure - it recognises that the overall sentiment is negative due to the contrast and the final negative clause. 
This demonstrates how contextual understanding allows Hugging Face to handle mixed sentiment correctly, while the bag-of-words model cannot.

In summary, the models agree on simple or strongly polarised sentences 
but diverge when the sentiment depends on word order词序, contrast对比, or subtle context上下文 - areas where classical models inherently fail.总而言之，对于简单或极化强烈的句子，两种模型的结果一致；但当情感取决于词序、对比或微妙的上下文时，模型结果则出现分歧——而这正是传统模型固有的缺陷所在。
"""
