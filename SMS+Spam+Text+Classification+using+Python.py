
# coding: utf-8

# In[3]:

import nltk


# In[4]:

#Step 1 - Data Collection

nltk.download()


# In[5]:

#Download dataset from UCI dataset for SMS Spam Collection - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection


# In[7]:

messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]


# In[8]:

len(messages)


# In[9]:

for num, message in enumerate(messages[:10]):
    print(num, message)
    print('\n')


# ![image.png](attachment:image.png)

# In[10]:

import pandas


# In[11]:

messages = pandas.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names = ['labels', 'message'])


# In[12]:

messages.head()


# In[13]:

#Step 2 - Basic Exploratory Data Analysis
messages.describe()


# In[14]:

messages.info()


# In[15]:

messages.groupby('labels').describe()


# In[16]:

#Feature Engineering

messages['length'] = messages['message'].apply(len)
messages.head()


# In[17]:

#Data Visualization

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[18]:

messages['length'].plot(bins=50, kind = 'hist')


# In[19]:

messages['length'].describe()


# In[20]:

messages[messages['length'] == 910]['message'].iloc[0]


# In[21]:

messages.hist(column='length', by='labels', bins = 50, figsize=(10,4))


# In[22]:

#Step 3 - Text Pre-processing
import string


# In[23]:

mess = 'Sample Message! Notice: It has punctuation'


# In[24]:

nopunc = [char for char in mess if char not in string.punctuation]
nopunc


# In[25]:

nopunc = ''.join(nopunc)


# In[26]:

nopunc


# In[27]:

from nltk.corpus import stopwords


# In[28]:

stopwords.words('english')[0:10]


# In[29]:

nopunc.split()


# In[30]:

clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[31]:

clean_mess


# In[32]:

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[33]:

messages.head()


# In[34]:

messages['message'].head(5).apply(text_process)


# In[35]:

messages.head(5)


# In[36]:

#Step 4 - Vectorization
from sklearn.feature_extraction.text import CountVectorizer


# In[37]:

bow_transformer = CountVectorizer(analyzer=text_process)


# In[38]:

bow_transformer.fit(messages['message'])


# In[39]:

message4 = messages['message'][3]
print(message4)


# In[40]:

bow4 = bow_transformer.transform([message4])
print(bow4)


# In[42]:

print (bow_transformer.get_feature_names()[9554])


# In[49]:

messages_bow = bow_transformer.transform(messages['message'])


# In[46]:

print ('Shape of Sparse Matrix: ', messages_bow.shape)
print ('Amount of Non-Zero occurences: ', messages_bow.nnz)
print ('sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1])))


# In[50]:

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)


# In[51]:

tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)


# In[52]:

print(tfidf_transformer.idf_[bow_transformer.vocabulary_['u']])


# In[53]:

messages_tfidf = tfidf_transformer.transform(messages_bow)


# In[55]:

print(messages_tfidf.shape)


# In[56]:

#Step 5 - Training a model
from sklearn.naive_bayes import MultinomialNB


# In[57]:

spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['labels'])


# In[59]:

print('Predicted: ', spam_detect_model.predict(tfidf4)[0])
print('Expected: ', messages['labels'][3])


# In[61]:

#Step 6 - Model Evaluation
all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)


# ![image.png](attachment:image.png)

# In[62]:

from sklearn.metrics import classification_report
print(classification_report(messages['labels'],all_predictions))


# In[63]:

from sklearn.cross_validation import train_test_split

msg_train, msg_tst, label_train, label_test = train_test_split(messages['message'], messages['labels'], test_size=0.2)


# In[64]:

print(len(msg_train), len(msg_tst), len(msg_train) + len(msg_tst))


# In[65]:

#Step 7: Creating a Data Pipeline
from sklearn.pipeline import Pipeline


# In[66]:

pipeline = Pipeline([('bow', CountVectorizer(analyzer=text_process)),
                    ('tfidf', TfidfTransformer()),
                    ('classifier', MultinomialNB())])


# In[67]:

pipeline.fit(msg_train, label_train)


# In[68]:

predictions = pipeline.predict(msg_tst)


# In[69]:

print(classification_report(predictions, label_test))


# In[ ]:



