# TF实战代码

from sklearn.feature_extraction.text import CountVectorizer

corpus = ['This is the first document',
          'This is the second document document',
          'tird one one one',
          'fourth one',
          'fifth one']

vectorizer = CountVectorizer(stop_words=None)
x = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names(), vectorizer.get_stop_words())
print(x.toarray())