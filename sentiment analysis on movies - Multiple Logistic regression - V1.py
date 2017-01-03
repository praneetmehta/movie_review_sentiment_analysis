#0 = bad
#1 = somewhat bad
#2 = neutral
#3 = somewhat good
#4 = good


import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error as msq
from nltk import word_tokenize, pos_tag
from nltk.stem.wordnet import WordNetLemmatizer


# %%
df= pd.read_csv('train.tsv', delimiter='\t')[['Phrase','Sentiment']]

# %%

wordnet_tags = ['n', 'v']
def lemmatize(token, tag):
    if tag[0].lower() in ['n', 'v']:
        return lemmatizer.lemmatize(token, tag[0].lower())
    return token
lemmatizer = WordNetLemmatizer()

def cleanText(text):
    text = re.sub("[^a-zA-Z]"," ",text)
    text = re.sub("/\r?\n|\r/"," ",text).lower()
    return text
    
# %%

df_copy = df.copy()
X = df_copy['Phrase'][(df['Sentiment'] != 2) | ((df['Sentiment'] == 2) & (df['Phrase'].str.split(' ').apply(len) < 10))]
Y = df_copy['Sentiment'][(df['Sentiment'] != 2) | ((df['Sentiment'] == 2) & (df['Phrase'].str.split(' ').apply(len) < 10))]

X_clean = [cleanText(train_text) for train_text in X]
tagged_X = [pos_tag(word_tokenize(document)) for document in X_clean]
tagged_X_lemm = [[lemmatize(token, tag) for token, tag in document] for document in tagged_X]
joined_lemm_X = [' '.join(text) for text in tagged_X_lemm]  
#split into train and test data
X_train_clean , X_test_clean, Y_train, Y_test = train_test_split(joined_lemm_X,Y)


# %%
plt.plot(range(0,len(X_train_clean)),sorted([len(string.split(' ')) for string in X_train_clean]))

# %%
### WITHOUT USING GRID SEARCH ###
vectorizer = TfidfVectorizer(stop_words = 'english', max_df=0.25, ngram_range=(1,2),use_idf = False)
X_train = vectorizer.fit_transform(X_train_clean)
X_test = vectorizer.transform(X_test_clean)
X_vec = vectorizer.transform(X)

regressor = LogisticRegression(C=10)
regressor.fit(X_train, Y_train)

#create confusion matrix
confusion_matrix = confusion_matrix(Y_test, regressor.predict(X_test))
print(confusion_matrix)

# %%
#plot and show the confusion matrix with color bar
plt.matshow(confusion_matrix)
plt.title('Sentiment Analysis from reviews')
plt.ylabel('True Values')
plt.xlabel('Predicted Values')
plt.colorbar()
plt.show()

# %%
#recall precision accuracy and f1 scores
print('Accuracy: %s'%accuracy_score(Y_test, regressor.predict(X_test)))
#print('Recall: %s'%recall_score(Y_test, regressor.predict(X_test), average='macro'))
#print('Precision: %s'%precision_score(Y_test, regressor.predict(X_test), average='macro'))
#print('F1: %s'%f1_score(Y_test, regressor.predict(X_test), average='macro'))
print('CR: %s'%classification_report(Y_test, regressor.predict(X_test)))
print('R Square: %s'%regressor.score(X_test, Y_test))
print('Mean sqared error: %s'%msq(Y_test, regressor.predict(X_test)))

### USING GRID SEARCH ### (called only when the funtion main is called)
# %%
#CROSS VAL SCORE
#print('Cross Val Score: %s'%cross_val_score(regressor, X_vec, Y, cv=5))

# %%
def main():
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(stop_words = 'english')),
        ('reg', LogisticRegression())
    ])
    parameters = {
        'vect__max_df': (0.25, 0.5, 0.75),
        'vect__ngram_range': ((1, 1), (1, 2), (1,3)),
        'vect__use_idf': (True, False)
    }
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1,verbose=1, scoring='accuracy')
    grid_search.fit(X_train_clean, Y_train)
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
    print('CR: %s'%classification_report(Y_test, grid_search.predict(X_test_clean)))
    print('R Square: %s'%grid_search.score(X_test_clean, Y_test))
    print('Mean sqared error: %s'%msq(Y_test, grid_search.predict(X_test_clean)))

#if __name__ == '__main__':
#    main()
# %%
def predictor(reviewText):
    
    
    score = (regressor.predict(vectorizer.transform([reviewText])))[0]
    if score == 0:
        return "1 STAR"
    elif score == 1:
        return "2 STAR"
    elif score == 2:
        return "3 STAR"
    elif score == 3:
        return "4 STAR"
    elif score == 4:
        return "5 STAR"


# %%
review = cleanText(open('./Samples/5-1','r').read())
print(predictor(review))


