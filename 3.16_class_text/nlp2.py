from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

#build a ml pipeline

corpus = [
    'i love the book',
    'this book was not so great',
    'the fit was great',
    'i love the shoes'
]

books ='Books'
clothing = 'Clothing'

categories = [books, books, clothing, clothing] #代表每一行是属于什么类别

vectorizer = CountVectorizer(ngram_range=(1,2)) #向量

vectors = vectorizer.fit_transform(corpus) #训练模型时：把文字转成文字+向量数字

print(vectorizer.get_feature_names_out())
#['book' 'fit' 'great' 'love' 'not' 'shoes' 'so' 'the' 'this' 'was']

#['book' 'book was' 'fit' 'fit was' 'great' 'love' 'love the' 'not'
# 'not so' 'shoes' 'so' 'so great' 'the' 'the book' 'the fit' 'the shoes'
# 'this' 'this book' 'was' 'was great' 'was not']

#we don't use stopwords here 
print(vectors.toarray())
#[[1 0 0 1 0 0 0 1 0 0]
# [1 0 1 0 1 0 1 0 1 1]
# [0 1 1 0 0 0 0 1 0 1]
# [0 0 0 1 0 1 0 1 0 0]]

#[[1 0 0 0 0 1 1 0 0 0 0 0 1 1 0 0 0 0 0 0 0]
# [1 1 0 0 1 0 0 1 1 0 1 1 0 0 0 0 1 1 1 0 1]
# [0 0 1 1 1 0 0 0 0 0 0 0 1 0 1 0 0 0 1 1 0]
# [0 0 0 0 0 1 1 0 0 1 0 0 1 0 0 1 0 0 0 0 0]]


clf = SVC(kernel='linear') #Support Vector Classification

#vectors向量 will be X
#send categories will be Y
clf.fit(vectors, categories) #use fit to train machine learning model




test_corpus = [
    'i love read this',
    'such a nice hat',
    'what a great book'
]

test_categories = [books, clothing, books] #加这行是为了检测accuracy

test_x = vectorizer.transform(test_corpus) #test测试模型时：only get vectors 向量没有文字

print(clf.predict(test_x)) #['Clothing' 'Clothing' 'Books'] 他把第一行看错了 看成了属于 clothing这个类别，第二行猜对了是属于 clothing，第三行也对了是属于 books的类别
print(clf.score(test_x, test_categories)) #0.6666666666666666 means 66% correct
