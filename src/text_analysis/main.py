from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

################################################################################
# Loading Dataset

categories = [
    'alt.atheism',
    'soc.religion.christian',
    'comp.graphics',
    'sci.med'
]

twenty_train = fetch_20newsgroups(
    subset='train',
    categories=categories,
    shuffle=True,
    random_state=42
)

print(twenty_train.target_names)

# twenty_train.data is a list of string
# each element is e-mail, contains header and body
print(len(twenty_train.data))

# twenty_train.filenames is a list of string
print(len(twenty_train.filenames))
print(twenty_train.filenames[:4])

# Header of first e-mail
print("\n".join(twenty_train.data[0].split("\n")[:3]))
print(twenty_train.target_names[twenty_train.target[0]])
print(twenty_train.target[:10])

for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])

################################################################################
# Tokenizing

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
print(X_train_counts.shape)

print(count_vect.vocabulary_.get(u'algorithm'))

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_if = tf_transformer.transform(X_train_counts)
print(X_train_if.shape)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

################################################################################
# Training Classifier
################################################################################

clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
docs_new = ['God is Love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))

################################################################################
# Building Pipeline
################################################################################

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])
text_clf.fit(twenty_train.data, twenty_train.target)

################################################################################
# Evaluation of the Performance on the test set
################################################################################

twenty_test = fetch_20newsgroups(
    subset='test',
    categories=categories,
    shuffle=True,
    random_state=12
)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))

# Test Support Vector Machine
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42,
                                           max_iter=5, tol=None))])
text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))

print(
    metrics.classification_report(
        twenty_test.target,
        predicted,
        target_names=twenty_test.target_names
    )
)

print(metrics.confusion_matrix(twenty_test.target, predicted))

################################################################################
# Evaluation of the Performance on the test set
################################################################################

