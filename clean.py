# -*- coding: utf-8 -*-
"""
Created on Sun May  6 19:25:49 2018

@author: lxb
"""

import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
cols = ['sentiment','id','date','query_string','user','text']
df = pd.read_csv("training.1600000.processed.noemoticon.csv",header=None, names=cols, encoding='latin-1')
# above line will be different depending on where you saved your data, and your file name
df.head()

df.sentiment.value_counts()

df.drop(['id', 'date', 'query_string','user'], axis = 1, inplace = True)
df[df.sentiment == 0].head(10)

df['pre_clean_len'] = [len(t) for t in df.text]


from pprint import pprint
data_dict = {
        'sentiment':{
                'type':df.sentiment.dtype,
                'description':'sentiment class - 0:negative, 1:positive'
         },
         'text':{
                 'type':df.text.dtype,
                 'description':'tweet text'
         },
         'pre_clean_len':{
                 'type':df.pre_clean_len.dtype,
                 'description':'Length of the tweet before cleaning'
         },
         'dataset_shape':df.shape
}

pprint(data_dict)


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize = (5, 5))
plt.boxplot(df.pre_clean_len)
plt.show()



df[df.pre_clean_len > 140].head(10)

from bs4 import BeautifulSoup
example1 = BeautifulSoup(df.text[279], 'lxml')
print(example1.get_text())


import re
re.sub(r'@[A-Za-z0-9]+','',df.text[343])

df.text[0]

re.sub('https?://[A-Za-z0-9./]+','',df.text[0])


testing = df.text[226].decode("utf-8-sig")
testing


re.sub("[^a-zA-Z]", " ", df.text[175])



from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))
def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()
testing = df.text[:100]
test_result = []
for t in testing:
    test_result.append(tweet_cleaner(t))
test_result

nums = [0,400000,800000,1200000,1600000]
print("Cleaning and parsing the tweets...\n")
clean_tweet_texts = []
for i in range(nums[0],nums[1]):
    if( (i+1)%10000 == 0 ):
        print("Tweets %d of %d has been processed" % ( i+1, nums[1] ))                                                                    
    clean_tweet_texts.append(tweet_cleaner(df['text'][i]))
    
    
clean_df = pd.DataFrame(clean_tweet_texts,columns=['text'])
clean_df['target'] = df.sentiment
clean_df.head()

clean_df.to_csv('clean_tweet.csv',encoding='utf-8')
csv = 'clean_tweet.csv'
my_df = pd.read_csv(csv,index_col=0)
my_df.head()