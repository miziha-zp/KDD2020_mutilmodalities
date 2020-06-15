#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import base64
from tqdm.notebook import tqdm 
from scipy.spatial.distance import cdist


# In[6]:


train=pd.read_csv('../../data/train/train.tsv',sep='\t',quoting=3,usecols=['query'])
train=pd.DataFrame(list(set(train['query'].tolist())),columns=['query'])
train.head(5)


# In[7]:


# 对词进行分割
train['split_query']=train['query'].apply(lambda x:x.split(' '))
train.head(5)


# In[8]:


# 最后一个词
# 最后一个词里面有些会出现 ‘-’ 需要分割

# 1、 十字绣 改一下
# 2、 tshirt 改一下
# 3、o-o- pump- 改一下

def fun(x):
    # 修正一下
    if x[-1]=='cross-stitch':
        x.pop(-1)
        x.append('cross') 
        x.append('embroidery')
    elif x[-1]=='t-shirt':
        x[-1]='shirt'
    elif x[-1]=='o-o-':
        x.pop(-1)
    elif x[-1]=='pump-':
        x[-1]='pump'
    
    
    if '-' in x[-1]:
        ans=x[-1].split('-')[-1]
        if len(ans)<3:  # 奇怪的东西  比如 o-o- pump-
            print(x,ans)
    else:
        ans=x[-1]
    return ans 

train['last_query_word']=train['split_query'].apply(fun)


# In[12]:


# 加载词向量
from gensim.models.keyedvectors import KeyedVectors
gensim_model = KeyedVectors.load_word2vec_format('../../external_resources/GoogleNews-vectors-negative300.bin', binary=True)


# # 计算了最后一个词的相似度

# In[12]:


# 在dataframe上操作unique query last word 最后merge
unique_last_query_word=pd.DataFrame(unique_last_query_word,columns=['last_query_word'])

# 寻找出 word2vec
unique_last_query_word['last_word_embedding']=np.empty((len(unique_last_query_word), 0)).tolist()
unique_last_query_word['not_find_word']=0

for i in tqdm(range(len(unique_last_query_word))):
    try:
        unique_last_query_word.at[i,'last_word_embedding']=list(gensim_model[unique_last_query_word.loc[i,'last_query_word']])
    except:
        unique_last_query_word.at[i,'last_word_embedding']=list(np.zeros((300, ), dtype='float32'))
        unique_last_query_word.at[i,'not_find_word']=1
        print(unique_last_query_word.loc[i,'last_query_word'])
print(len(unique_last_query_word[unique_last_query_word['not_find_word']==1]))

# 这里去掉了找不到的单词,最终在merge的时候都被去掉了【大概去掉了5k条数据】
unique_last_query_word=unique_last_query_word[unique_last_query_word['not_find_word']==0]   
unique_last_query_word


# In[13]:


# 计算相互之间距离
distance=cdist(np.asarray(unique_last_query_word['last_word_embedding'].tolist()),np.asarray(unique_last_query_word['last_word_embedding'].tolist()))
print('distance calculating finish')
#去掉了自己的最近距离,剩下的按照index排序，   只取前50个相似
unique_last_query_word['last_word_similarity_index']=[list(x) for x in np.argsort(distance,axis=1)[:,1:50]]   
unique_last_query_word.reset_index(drop=True,inplace=True)
unique_last_query_word.head(5)


# In[14]:


# index 转化为last word
unique_last_query_word['last_word_similarity']=np.empty((len(unique_last_query_word), 0)).tolist()
for i in tqdm(range(len(unique_last_query_word))):
    unique_last_query_word.at[i,'last_word_similarity']=[unique_last_query_word.loc[x,'last_query_word'] for x in unique_last_query_word.loc[i,'last_word_similarity_index']]
unique_last_query_word.head(5)


# In[15]:


train=pd.merge(train,unique_last_query_word[['last_query_word','last_word_similarity']],on='last_query_word')
train.head(5)


# In[16]:


train.to_pickle("../../user_data/hard_sample_data/transfer_query.pkl")


# # query 相似度

# In[10]:


train=pd.read_pickle("../../user_data/hard_sample_data/transfer_query.pkl")
print(train.shape)


# In[18]:


count=0
def avg_feature_vector(sentence, model=gensim_model, num_features=300):  #输入 句子  返回 向量（所有词的平均）
    global count 
    count=count+1
    if count%10000==0:
        print('*'*5,sentence,count)
    words = sentence.split()
    #feature vector is initialized as an empty array
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in model.index2word:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return list(feature_vec)


# In[19]:


# 获取每个query的word2vec
print('query word2vec starting')
train['average_vector']=train['query'].apply(lambda x:avg_feature_vector(x,gensim_model,num_features=300))  
print('query word2vec finish')


# In[20]:


train.to_pickle("../../user_data/hard_sample_data/transfer_query_add_query_embedding.pkl")


# 相似度

# In[5]:


train=pd.read_pickle("../user_data/hard_sample_data/transfer_query_add_query_embedding.pkl")
train.head(5)


# 可能是因为数据太多了（部分比如说book有9w条）      
# 这边last word 少的就增加到1k，多于1w的就分割到1w

# In[7]:


train['new_last_word']=train['last_query_word']
train.head(5)


# In[10]:


# query 长度 哈希表
lengthhash={}
for index,data in tqdm(train.groupby(['last_query_word'])):
#     print(len(data))
    lengthhash[index]=len(data)


# In[11]:


count=0
def func(x):
    if len(x)<1000:  # 如果数据<400 归类
        for i in x['last_word_similarity'].tolist()[0]:# 获取当前最后一个词   最接近的词 并将其归为这一类 
            if lengthhash[i]>1500:
                x['new_last_word']=i                  # 将这一类抹掉 赋予新的类别 最想近的类别
                break
            else:
                x['new_last_word']='bujidao'                  # 将这一类抹掉 赋予新的类别 最想近的类别
    return x

train=train.groupby(['last_query_word']).apply(func)
train['new_last_word'].value_counts()  # 处理之后的效果


# In[13]:


def func(x):
#     print(x['last_query_word'])
    MAXLENGTH=3000    # 如果数据》5k条 拆分
    SPLITLENGTH=2500  # 超过的按照4k分割
    if len(x)>MAXLENGTH:  
        now_last_query_word=x['new_last_word'].tolist()[0]  # 获取当前最后一个词      
        for i in range(0,len(x),SPLITLENGTH):
            length=len(x.iloc[i:i+SPLITLENGTH,:])
            x.iloc[i:i+SPLITLENGTH,5]=[now_last_query_word+'_'+str(i) for x in range(length)]  #第五个是new last word
    
    return x
train=train.groupby(['new_last_word']).apply(func)
train['new_last_word'].value_counts()  # 处理之后的效果


# In[15]:


from scipy.spatial.distance import pdist,squareform
# train['inner_group_similarity_value'] = np.empty((len(train), 0)).tolist()
# train['inner_group_similarity_index'] = np.empty((len(train), 0)).tolist()
train['all_index_similarity_index'] = np.empty((len(train), 0)).tolist()
count=0


# In[18]:


count=0
def fun(df):
    global count 
    count=count+1
    if count%12==0:
        print('*'*5,count/631)
#     print('*'*5,count/696)
    distance=squareform(pdist(np.asarray(df['average_vector'].tolist())))
#     df['inner_group_similarity_value']=[list(x) for x in distance]   # 按照group内index排序的相似性
#     df['inner_group_similarity_index']=[list(x) for x in np.argsort(distance,axis=1)[:,1:1000]]   # 排列之后的  这种算出来的index是group内的index  按照顺序
    
    df['all_index_similarity_index']=[list([df.index[i] for i in g]) for g in np.argsort(distance,axis=1)[:,1:1000]] # 矫正之后的index才是整个数据集上的index
    # 后面和数据集合并的时候把这个index当成值，取出来
    return df
train=train.groupby(['new_last_word']).apply(fun)


# In[ ]:


hasmhap={}
for i in tqdm(range(len(df))):
    hashmap[i]=df.at[i,'query']


# In[ ]:


import random
train['n_query'] = np.empty((len(train), 0)).tolist()
def funx(df):
    MAX=int(0.8*len(df))+10
    MIN=int(0.8*len(df))
    ans=[hashmap[i] for i in df['all_index_similarity_index'][MIN:MAX]]
    df.loc[:,'n_query']=ans
    if len(ans)<=8:
        df.loc[:,'n_query']=[random.choice(range(10)) for i in range(10)]
    return df
train.groupby('new_last_word').apply(funx)


# In[ ]:


train=train[['query','n_query']]


# In[ ]:


data = []
for entry in tqdm(df.values):
    query, n_query = entry
    data.append({
        "query": query,
        "n_query": n_query,
    })
pickle.dump(data, open('../../user_data/hard_sample_data/n_query_hashmap.pkl', 'wb'))

