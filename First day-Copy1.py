#!/usr/bin/env python
# coding: utf-8

# In[92]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn 
import scipy as sk


# In[ ]:





# In[ ]:





# In[93]:


df=pd.read_csv(r'C:\Users\DELL\Desktop\mera data.csv')


# In[94]:


df['index']=df['Unnamed: 0']


# In[146]:


pd.set_option('max_row',None)


# In[96]:



from sklearn.feature_extraction.text import CountVectorizer 


# In[97]:


from sklearn.metrics.pairwise import cosine_similarity


# In[98]:


feat =['title','genres','imdb_rating']
 


# In[99]:



df['comb_feat']=df['title']+" "+df['genres']


# In[100]:


df['comb_feat']


# In[101]:


movie_user_like ="Uri: The Surgical Strike"


# In[102]:


def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]
    
def get_index_from_title(title):
    return df[df.title == title]["Unnamed: 0"].values[0]
    

movie_index= get_index_from_title(movie_user_like)


# In[103]:


cv=CountVectorizer()


# In[104]:


count_matrix=cv.fit_transform(df['comb_feat'])


# In[105]:


movie_index


# In[106]:


cos_sim=cosine_similarity(count_matrix)


# In[107]:


cos_sim


# In[108]:


similar_mov=list(enumerate(cos_sim[movie_index]))


# In[109]:


similar_mov


# In[110]:


sorted_simliar_mov= sorted(similar_mov,key=lambda x:x[1],reverse=True)


# In[111]:


sorted_simliar_mov


# In[112]:



for movie in sorted_simliar_mov:
    print(get_title_from_index(movie[0]))


# In[124]:


df


# In[157]:


user_mood= input(' ENTER YOUR MOOD ')


# In[158]:


if user_mood == 'sad':
    for each in ['Drama']:
        sub=each
        df['index']=df['genres'].str.find(sub)
    
if user_mood == 'happy':
    for each in ['Action','Comedy','Musical']:
        sub=each
        df['index']=df['genres'].str.find(sub)
if user_mood == 'Excited':
     for each in ['Action','Romance']:
        sub=each
        df['index']=df['genres'].str.find(sub)
if user_mood == 'Anticipation':
     for each in ['Crime','War']:
        sub=each
        df['index']=df['genres'].str.find(sub)
if user_mood == 'Anger':
     for each in ['Family','Musical','Comedy']:
        sub=each
        df['index']=df['genres'].str.find(sub)
if user_mood == 'Depressing':
     for each in ['Drama','Biography']:
        sub=each
        df['index']=df['genres'].str.find(sub)
if user_mood == 'Confusing':
     for each in ['Thriller','Fantasy','Crime']:
        sub=each
        df['index']=df['genres'].str.find(sub)
if user_mood == 'Inspiring':
     for each in ['Biography','Documentary','Sport','War']:
        sub=each
        df['index']=df['genres'].str.find(sub)
if user_mood == 'Thrilling':
     for each in ['Horror','Mystery']:
        sub=each
        df['index']=df['genres'].str.find(sub)


# In[160]:


print((df[df['index']>=0]['title']).unique())


# In[150]:


feat=df['title'].unique()


# In[154]:


pd.set_option('max_row',None)
print(feat)


# In[156]:


df.drop_duplicates(subset='title',keep=False)


# In[ ]:




