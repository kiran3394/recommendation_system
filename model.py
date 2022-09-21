#!/usr/bin/env python
# coding: utf-8

# In[214]:


import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from IPython import get_ipython

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from collections import Counter
from imblearn.over_sampling import SMOTE 
from sklearn import preprocessing
import pickle
from IPython.display import Image  
from six import StringIO
from sklearn.tree import export_graphviz
import pydotplus, graphviz
import random


# In[215]:


df=pd.read_csv('sample30.csv')
df.head()


# In[216]:


df.info()


# In[217]:


df.isnull().sum()


# In[218]:


df['reviews_text']


# In[219]:


#Remove all rows where reviews_username column is nan
df.dropna(subset=['reviews_text'], inplace=True)
df.dropna(subset=['reviews_username'], inplace=True)


# In[220]:


df.isnull().sum()


# In[221]:


df.info()


# In[222]:


df['user_sentiment'].value_counts()


# In[223]:


sns.countplot(x=df['user_sentiment'])
plt.show()


# In[224]:


df['reviews_rating'].value_counts()


# In[225]:


sns.countplot(x=df['reviews_rating'])
plt.show()


# In[226]:


from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
).generate(str(data))

    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(df['reviews_text'])


# In[227]:


positivereviews=df["reviews_text"].loc[df["user_sentiment"]=='Positive']
print(positivereviews)


# In[228]:


show_wordcloud(positivereviews)


# In[229]:


negativereviews=df["reviews_text"].loc[df["user_sentiment"]=='Negative']
print(negativereviews)


# In[230]:


show_wordcloud(negativereviews)


# In[231]:


# calling the label encoder function
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'user_sentiment'. 
df["user_sentiment"]= label_encoder.fit_transform(df['user_sentiment']) 
  
df['user_sentiment'].unique() 


# In[232]:


#Text PreProcessing


# In[233]:


## Converting the read dataset in to a list of tuples, each tuple(row) contianing the message and it's label
data_set = []
for index,row in df.iterrows():
    data_set.append((row['reviews_text'], row['user_sentiment']))
print(data_set[:5])
print(len(data_set))


# In[234]:


import re
import nltk


# In[235]:


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


# In[236]:


from bs4 import BeautifulSoup

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


# In[237]:


from tqdm import tqdm
processed_reviews = []
processed_reviews_with_label = []
# tqdm is for printing the status bar

for (sentance,label) in tqdm(data_set):
    sentance = re.sub(r"http\S+", "", sentance)
    sentance = BeautifulSoup(sentance, 'lxml').get_text()
    sentance = decontracted(sentance)
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    word_list=nltk.word_tokenize(sentance)
    sentance = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    processed_reviews.append(sentance.strip())
        
    processed_reviews_with_label.append((sentance.strip(),label))


# In[238]:


processed_reviews_with_label


# In[239]:


#Feature Extraction


# In[240]:


tf_idf_vect = TfidfVectorizer(ngram_range=(1,1),stop_words='english',lowercase=True, min_df=10)
tf_idf_vect.fit(processed_reviews)
print("some sample features(unique words in the corpus)",tf_idf_vect.get_feature_names()[0:10])
print('='*50)

final_tf_idf = tf_idf_vect.transform(processed_reviews)
print("the type of count vectorizer ",type(final_tf_idf))
print("the shape of out text TFIDF vectorizer ",final_tf_idf.get_shape())
print("the number of unique words ", final_tf_idf.get_shape()[1])


# In[261]:


tfidf=pd.DataFrame(final_tf_idf.toarray(),columns=tf_idf_vect.get_feature_names())
tfidf


# In[262]:


filename='tf_idf_model.pkl'
pickle.dump(tf_idf_vect,open(filename,'wb'))


# In[263]:


word_features=tf_idf_vect.get_feature_names()
word_features


# In[260]:


len(word_features)


# In[244]:


## - creating slicing index at 80% threshold
sliceIndex = int((len(data_set)*.8))


# In[245]:


## - shuffle the pack to create a random and unbiased split of the dataset
random.shuffle(data_set)


# In[246]:


train_messages, test_messages = data_set[:sliceIndex], data_set[sliceIndex:]


# In[247]:


train_messages


# In[248]:


print(len(train_messages))
print(len(test_messages))


# In[249]:


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


# In[250]:


X_train =  pd.DataFrame([extract_features(sentence) for (sentence,label) in train_messages])
X_test =  pd.DataFrame([extract_features(sentence) for (sentence,label) in test_messages])
Y_train =  pd.DataFrame([label for (sentence,label) in train_messages])
Y_test =  pd.DataFrame([label for (sentence,label) in test_messages])


# In[251]:


X_train = X_train.astype(float)
Y_train = Y_train.astype(float)
X_test = X_test.astype(float)
Y_test = Y_test.astype(float)


# In[252]:


print('Training set size : ', len(X_train))
print('Test set size : ', len(X_test))


# In[173]:


#creating the objects
logreg_cv = LogisticRegression(random_state=0)
rf_cv=RandomForestClassifier()
nb_cv=BernoulliNB()
cv_dict = {0: 'Logistic Regression', 1: 'Random Forest',2:'Naive Bayes'}
cv_models=[logreg_cv,rf_cv,nb_cv]

for i,model in enumerate(cv_models):
    print("{} Test Accuracy: {}".format(cv_dict[i],cross_val_score(model,X_train,Y_train.values.ravel(), cv=10, scoring ='accuracy').mean()))


# In[253]:


rf_model=rf_cv.fit(X_train,Y_train.values.ravel())


# In[254]:


filename='sentiment_analysis_model.pkl'
pickle.dump(rf_model,open(filename,'wb'))


# In[277]:


#User based Recommendation System


# In[393]:


train, test = train_test_split(df, test_size=0.30, random_state=31)


# In[394]:


print(train.shape)
print(test.shape)


# In[492]:


df_pivot=train.pivot_table(index='reviews_username',columns='name',values='reviews_rating',aggfunc='mean').fillna(0)
df_pivot.head(3)


# In[493]:


# Copy the train dataset into dummy_train
dummy_train = train.copy()


# In[494]:


# The product not rated by user is marked as 1 for prediction. 
dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)


# In[495]:


# Convert the dummy train dataset into matrix format.
dummy_train = dummy_train.pivot_table(index='reviews_username',columns='name',values='reviews_rating',aggfunc='mean').fillna(1)


# In[496]:


dummy_train.head()


# In[497]:


from sklearn.metrics.pairwise import pairwise_distances

# Creating the User Similarity Matrix using pairwise_distance function.
user_correlation = 1 - pairwise_distances(df_pivot, metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation)


# In[498]:


user_correlation.shape


# In[499]:


df_pivot=train.pivot_table(index='reviews_username',columns='name',values='reviews_rating',aggfunc='mean')
df_pivot.head(3)


# In[500]:


mean = np.nanmean(df_pivot, axis=1)
df_subtracted = (df_pivot.T-mean).T


# In[501]:


df_subtracted.head()


# In[502]:


# Creating the User Similarity Matrix using pairwise_distance function.
user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation)


# In[503]:


user_correlation[user_correlation<0]=0
user_correlation


# In[504]:


user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))
user_predicted_ratings


# In[505]:


user_predicted_ratings.shape


# In[506]:


user_final_rating = np.multiply(user_predicted_ratings,dummy_train)
user_final_rating.head()


# In[507]:


user_final_rating.head(2)


# In[508]:


# Take the user ID as input.
user_input = input("Enter your user name")


# In[635]:


d = user_final_rating.loc['frances'].sort_values(ascending=False)[0:20]
d.index.tolist()


# In[615]:


filename='recommendation_system_model.pkl'
pickle.dump(user_final_rating,open(filename,'wb'))


# In[511]:


# Find out the common users of test and train dataset.
common = test[test['reviews_username'].isin(train['reviews_username'])]
common.shape


# In[512]:


common_user_based_matrix=common.pivot_table(index='reviews_username',columns='name',values='reviews_rating',aggfunc='mean')


# In[513]:


# Convert the user_correlation matrix into dataframe.
user_correlation_df = pd.DataFrame(user_correlation)


# In[514]:


df_subtracted.head()


# In[515]:


user_correlation_df['reviews_username'] = df_subtracted.index
user_correlation_df.set_index('reviews_username',inplace=True)
user_correlation_df.head()


# In[516]:


list_name = common['reviews_username'].tolist()

user_correlation_df.columns = df_subtracted.index.tolist()


user_correlation_df_1 =  user_correlation_df[user_correlation_df.index.isin(list_name)]


# In[517]:


user_correlation_df_1.shape


# In[518]:


user_correlation_df_2 = user_correlation_df_1.T[user_correlation_df_1.T.index.isin(list_name)]


# In[519]:


user_correlation_df_3 = user_correlation_df_2.T


# In[520]:


user_correlation_df_3.head()


# In[521]:


user_correlation_df_3.shape


# In[522]:


user_correlation_df_3[user_correlation_df_3<0]=0

common_user_predicted_ratings = np.dot(user_correlation_df_3, common_user_based_matrix.fillna(0))
common_user_predicted_ratings


# In[523]:


dummy_test = common.copy()

dummy_test['reviews_rating'] = dummy_test['reviews_rating'].apply(lambda x: 1 if x>=1 else 0)

dummy_test = dummy_test.pivot_table(index='reviews_username', columns='name', values='reviews_rating').fillna(0)


# In[524]:


dummy_test.shape


# In[525]:


common_user_predicted_ratings = np.multiply(common_user_predicted_ratings,dummy_test)


# In[526]:


common_user_predicted_ratings.head(2)


# In[537]:


from sklearn.preprocessing import MinMaxScaler
from numpy import *

X  = common_user_predicted_ratings.copy() 
X = X[X>0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))

print(y)


# In[538]:


common_=common.pivot_table(index='reviews_username',columns='name',values='reviews_rating',aggfunc='mean')


# In[539]:


# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))


# In[541]:


rmse = (sum(sum((common_ - y )**2))/total_non_nan)**0.5
print(rmse)


# In[ ]:


#Item - Item based recommendation System


# In[542]:


df_pivot=train.pivot_table(index='reviews_username',columns='name',values='reviews_rating',aggfunc='mean').T
df_pivot.head(3)


# In[543]:


mean = np.nanmean(df_pivot, axis=1)
df_subtracted = (df_pivot.T-mean).T


# In[544]:


df_subtracted.head()


# In[545]:


# Item Similarity Matrix
item_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
item_correlation[np.isnan(item_correlation)] = 0
print(item_correlation)


# In[546]:


item_correlation[item_correlation<0]=0
item_correlation


# In[547]:


item_predicted_ratings = np.dot((df_pivot.fillna(0).T),item_correlation)
item_predicted_ratings


# In[548]:


item_predicted_ratings.shape


# In[549]:


dummy_train.shape


# In[550]:


item_final_rating = np.multiply(item_predicted_ratings,dummy_train)
item_final_rating.head()


# In[552]:


# Take the user ID as input
user_input = input("Enter your user name")
print(user_input)


# In[553]:


# Recommending the Top 5 products to the user.
d = item_final_rating.loc[user_input].sort_values(ascending=False)[0:20]
d


# In[554]:


common =  test[test['name'].isin(train['name'])]
common.shape


# In[555]:


common.head(4)


# In[556]:


common_item_based_matrix=common.pivot_table(index='reviews_username',columns='name',values='reviews_rating',aggfunc='mean').T


# In[557]:


common_item_based_matrix.shape


# In[558]:


item_correlation_df = pd.DataFrame(item_correlation)


# In[559]:


item_correlation_df['name'] = df_subtracted.index
item_correlation_df.set_index('name',inplace=True)
item_correlation_df.head()


# In[561]:


list_name = common['name'].tolist()
item_correlation_df.columns = df_subtracted.index.tolist()

item_correlation_df_1 =  item_correlation_df[item_correlation_df.index.isin(list_name)]


# In[562]:


item_correlation_df_2 = item_correlation_df_1.T[item_correlation_df_1.T.index.isin(list_name)]

item_correlation_df_3 = item_correlation_df_2.T


# In[563]:


item_correlation_df_3.head()


# In[564]:


item_correlation_df_3[item_correlation_df_3<0]=0

common_item_predicted_ratings = np.dot(item_correlation_df_3, common_item_based_matrix.fillna(0))
common_item_predicted_ratings


# In[565]:


common_item_predicted_ratings.shape


# In[567]:


dummy_test = common.copy()

dummy_test['reviews_rating'] = dummy_test['reviews_rating'].apply(lambda x: 1 if x>=1 else 0)

dummy_test = dummy_test.pivot_table(index='reviews_username', columns='name', values='reviews_rating').T.fillna(0)

common_item_predicted_ratings = np.multiply(common_item_predicted_ratings,dummy_test)


# In[568]:


common_ = common.pivot_table(index='reviews_username', columns='name', values='reviews_rating').T


# In[569]:


from sklearn.preprocessing import MinMaxScaler
from numpy import *

X  = common_item_predicted_ratings.copy() 
X = X[X>0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))

print(y)


# In[570]:


# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))


# In[571]:


rmse = (sum(sum((common_ - y )**2))/total_non_nan)**0.5
print(rmse)


# In[ ]:




