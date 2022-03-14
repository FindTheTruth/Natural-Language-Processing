# TF-IDF实战代码

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

corpus = ['This is the first document',
          'This is the second document document',
          'tird one one one',
          'fourth one',
          'fifth one']

# 转换方式1
vectorizer = CountVectorizer(stop_words=None)
x = vectorizer.fit_transform(corpus)
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(x)
print("--------------------------way one ------------------------")
print(tfidf)
print(tfidf.toarray())

# 转换方式2
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words=None)
x = vectorizer.fit_transform(corpus)
print("--------------------------way two ------------------------")
print(x)
print(x.toarray())