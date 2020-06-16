#!/usr/bin/env python
# coding: utf-8

# imgoral + hahaha 0.70622908099897

# In[1]:


import numpy as np
import pandas as pd
import swifter
import glob, base64
from pathlib import Path
import json

import lightgbm as lgb
from sklearn.model_selection import KFold

import zipfile
import os
from tqdm import tqdm 
def zipFile(filepath,outFullName):
    if not os.path.exists(filepath):
        print('not found')
    zip = zipfile.ZipFile(outFullName,"w",zipfile.ZIP_DEFLATED)
    zip.write(filepath, filepath)
    zip.close()
    
    
def get_top5_ans(testa_df,result_name):   
    sub = testa_df[['query_id','product_id',result_name]].sort_values(by=['query_id',result_name],ascending=[True,False])
    sub = sub.groupby('query_id').apply(lambda x: x.iloc[:5])[['query_id','product_id']].reset_index(drop=True)
    sub = sub.groupby('query_id')['product_id'].agg(list).reset_index()

    for i in range(1,6):
        sub[f'product{i}'] = sub['product_id'].apply(lambda x: x[i-1])
    sub = sub.rename(columns={'query_id':'query-id'})
    sub = sub[['query-id','product1','product2','product3','product4','product5']]
    return sub



# In[3]:




valid_df = pd.read_csv('../data/valid/valid.tsv', sep='\t')
# train_df = pd.read_csv(BASE_DIR/'train.sample.tsv', sep='\t')
testa_df = pd.read_csv('../data/testB/testB.tsv', sep='\t')

print('reading file finished')

# In[4]:


def parse_base64(df):
    def _decode_rows(row):
        row['boxes'] = np.frombuffer(base64.b64decode(row['boxes']), dtype=np.float32).reshape(row['num_boxes'], 4)
        row['features'] = np.frombuffer(base64.b64decode(row['features']), dtype=np.float32).reshape(row['num_boxes'], 2048)
        row['class_labels'] = np.frombuffer(base64.b64decode(row['class_labels']), dtype=np.int64).reshape(row['num_boxes'])
        return row
    df = df.swifter.apply(lambda x: _decode_rows(x), axis=1)
    return df

valid_df = parse_base64(valid_df)
# train_df = parse_base64(train_df)
testa_df = parse_base64(testa_df)

# data=pd.read_csv('../data/RAW_DATA/multimodal_labels.txt',sep='\t')
# multimodal_labels={}
# for i in range(len(data)):
#     multimodal_labels[data.loc[i,'category_id']]=data.loc[i,'category_name']
# multimodal_labels 
with open('../data/valid/valid_answer.json','r') as f:
    val_anser = json.loads(f.readlines()[0])
    
with open('../data/valid/valid_answer.json') as f:
    gt = json.load(f)

def get_label(x):
    query_id_ = x["query_id"]
    p_id = x["product_id"]
    if p_id in gt[str(query_id_)]:
        return 1
    else:
        return 0
valid_df["label"] = valid_df.apply(get_label,axis=1)

def parse_id_lst(x, i):
    if i >= len(x):
        return np.nan
    else:
        return x[i]


val_anser_df = pd.DataFrame(val_anser.items(), columns=[
                            'query_id', 'product_id_lst'])
for i in range(5):
    val_anser_df[f'product_id_{i}'] = val_anser_df['product_id_lst'].apply(
        lambda x: parse_id_lst(x, i))
val_anser_df = val_anser_df.drop('product_id_lst', axis=1).set_index(
    "query_id").stack().reset_index()
val_anser_df = val_anser_df.rename(
    columns={'level_1': 'rank', 0: 'product_id'})
val_anser_df['product_id'] = val_anser_df['product_id'].astype(int)
val_anser_df['query_id'] = val_anser_df['query_id'].astype(int)
# val_anser_df['rank_score'] = val_anser_df['rank'].apply(lambda x: 6-int(x[-1]))
val_anser_df['rank_score'] = 1
valid_df = valid_df.merge(val_anser_df[['product_id','query_id','rank_score']],on=['product_id','query_id'],how='left')
valid_df['rank_score'] = valid_df['rank_score'].fillna(0.5)
valid_df['rank_score'] = valid_df['rank_score'].replace(0.5,0).astype(int)

valid_df['features'] = valid_df['features'].swifter.apply(lambda x: x.mean(axis=0))
for i in tqdm(range(2048)):
    valid_df[f'features_{i}'] = valid_df['features'].apply(lambda x: x[i])

testa_df['features'] = testa_df['features'].swifter.apply(lambda x: x.mean(axis=0))
for i in tqdm(range(2048)):
    testa_df[f'features_{i}'] = testa_df['features'].apply(lambda x: x[i])

INDEX = ['product_id','query_id']


# beike
def parse_beike_oof(path_name,prob_name='prob'):
    query_id_list = []
    product_id_list = []
    prob_list = []
    file = open(path_name)
    all_data = file.readlines()
    for i in tqdm(all_data):
        query_id = int(i.split(",")[0])
        first_dot_index = 0
        for index,s in enumerate(i):
            if s == ',':
                first_dot_index = index
                break
                
        product_list_t = i[first_dot_index+3:-3]
#         print(product_list_t)
        for product_t in product_list_t.split("], ["):
            product_t_str = product_t
            prob = float(product_t_str.split(',')[0])
            product_id_t = int(product_t_str.split(',')[-1][2:-1])
            
            query_id_list.append(query_id)
            product_id_list.append(product_id_t)
            prob_list.append(prob)
#             print(prob)
#             print(product_id_t)
#         print(product_list_t)
    
    df = pd.DataFrame({
        "query_id":query_id_list,
        "product_id":product_id_list,
        prob_name:prob_list
    })
    
    print(df.head())
    return df

print("#"*20)
print("loading prob")
pro_df = parse_beike_oof(r"../user_data/embeding_data/beike/6894904689181448/test.txt","beike1")
testa_df = testa_df.merge(pro_df,how='left',on=['query_id','product_id'])
pro_df = parse_beike_oof(r"../user_data/embeding_data/beike/6894904689181448/val.txt","beike1")
valid_df = valid_df.merge(pro_df,how='left',on=['query_id','product_id'])

pro_df = parse_beike_oof(r"../user_data/embeding_data/beike/6862120525059028/test.txt",'beike2')
testa_df = testa_df.merge(pro_df,how='left',on=['query_id','product_id'])
pro_df = parse_beike_oof(r"../user_data/embeding_data/beike/6862120525059028/val.txt",'beike2')
valid_df = valid_df.merge(pro_df,how='left',on=['query_id','product_id'])

pro_df = parse_beike_oof(r"../user_data/embeding_data/beike/6738901694360588/test.txt",'beike3')
testa_df = testa_df.merge(pro_df,how='left',on=['query_id','product_id'])
pro_df = parse_beike_oof(r"../user_data/embeding_data/beike/6738901694360588/val.txt",'beike3')
valid_df = valid_df.merge(pro_df,how='left',on=['query_id','product_id'])

pro_df = parse_beike_oof(r"../user_data/embeding_data/beike/1/test.txt",'beike4')
testa_df = testa_df.merge(pro_df,how='left',on=['query_id','product_id'])
pro_df = parse_beike_oof(r"../user_data/embeding_data/beike/1/val.txt",'beike4')
valid_df = valid_df.merge(pro_df,how='left',on=['query_id','product_id'])

pro_df = parse_beike_oof(r"../user_data/embeding_data/beike/2/test.txt",'beike5')
testa_df = testa_df.merge(pro_df,how='left',on=['query_id','product_id'])
pro_df = parse_beike_oof(r"../user_data/embeding_data/beike/2/val.txt",'beike5')
valid_df = valid_df.merge(pro_df,how='left',on=['query_id','product_id'])

pro_df = parse_beike_oof(r"../user_data/embeding_data/beike/3/test.txt",'beike6')
testa_df = testa_df.merge(pro_df,how='left',on=['query_id','product_id'])
pro_df = parse_beike_oof(r"../user_data/embeding_data/beike/3/val.txt",'beike6')
valid_df = valid_df.merge(pro_df,how='left',on=['query_id','product_id'])


pro_df = parse_beike_oof(r"../user_data/embeding_data/beike/1/test_32.txt",'beike7')
testa_df = testa_df.merge(pro_df,how='left',on=['query_id','product_id'])
pro_df = parse_beike_oof(r"../user_data/embeding_data/beike/1/val_32.txt",'beike7')
valid_df = valid_df.merge(pro_df,how='left',on=['query_id','product_id'])

pro_df = parse_beike_oof(r"../user_data/embeding_data/beike/2/test_33.txt",'beike8')
testa_df = testa_df.merge(pro_df,how='left',on=['query_id','product_id'])
pro_df = parse_beike_oof(r"../user_data/embeding_data/beike/2/val_33.txt",'beike8')
valid_df = valid_df.merge(pro_df,how='left',on=['query_id','product_id'])

pro_df = parse_beike_oof(r"../user_data/embeding_data/beike/3/test_31.txt",'beike9')
testa_df = testa_df.merge(pro_df,how='left',on=['query_id','product_id'])
pro_df = parse_beike_oof(r"../user_data/embeding_data/beike/3/val_31.txt",'beike9')
valid_df = valid_df.merge(pro_df,how='left',on=['query_id','product_id'])


pro_df = parse_beike_oof(r"../user_data/embeding_data/beike/4/test.txt",'beike10')
testa_df = testa_df.merge(pro_df,how='left',on=['query_id','product_id'])
pro_df = parse_beike_oof(r"../user_data/embeding_data/beike/4/val.txt",'beike10')
valid_df = valid_df.merge(pro_df,how='left',on=['query_id','product_id'])

pro_df = parse_beike_oof(r"../user_data/embeding_data/beike/5/test.txt",'beike11')
testa_df = testa_df.merge(pro_df,how='left',on=['query_id','product_id'])
pro_df = parse_beike_oof(r"../user_data/embeding_data/beike/5/val.txt",'beike11')
valid_df = valid_df.merge(pro_df,how='left',on=['query_id','product_id'])

pro_df = parse_beike_oof(r"../user_data/embeding_data/beike/6/test.txt",'beike12')
testa_df = testa_df.merge(pro_df,how='left',on=['query_id','product_id'])
pro_df = parse_beike_oof(r"../user_data/embeding_data/beike/6/val.txt",'beike12')
valid_df = valid_df.merge(pro_df,how='left',on=['query_id','product_id'])


# In[7]:


pro_df = parse_beike_oof(r"../user_data/embeding_data/lcy_0609_n_sample/result_most_hard/test/test.txt",'lcy060901')
testa_df = testa_df.merge(pro_df,how='left',on=['query_id','product_id'])
pro_df = parse_beike_oof(r"../user_data/embeding_data/lcy_0609_n_sample/result_most_hard/val/val_10000.txt",'lcy060901')
valid_df = valid_df.merge(pro_df,how='left',on=['query_id','product_id'])


pro_df = parse_beike_oof(r"../user_data/embeding_data/lcy_0609_n_sample/result_n_query/test/test.txt",'lcy060902')
testa_df = testa_df.merge(pro_df,how='left',on=['query_id','product_id'])
pro_df = parse_beike_oof(r"../user_data/embeding_data/lcy_0609_n_sample/result_n_query/val/val_10000.txt",'lcy060902')
valid_df = valid_df.merge(pro_df,how='left',on=['query_id','product_id'])

pro_df = parse_beike_oof(r"../user_data/embeding_data/result_0610_most_hard/test/test.txt",'lcy061001')
testa_df = testa_df.merge(pro_df,how='left',on=['query_id','product_id'])
pro_df = parse_beike_oof(r"../user_data/embeding_data/result_0610_most_hard/val/val_10000.txt",'lcy061001')
valid_df = valid_df.merge(pro_df,how='left',on=['query_id','product_id'])

pro_df = parse_beike_oof(r"../user_data/embeding_data/result_0610_n_sample/test/test.txt",'lcy061002')
testa_df = testa_df.merge(pro_df,how='left',on=['query_id','product_id'])
pro_df = parse_beike_oof(r"../user_data/embeding_data/result_0610_n_sample/val/val_10000.txt",'lcy061002')
valid_df = valid_df.merge(pro_df,how='left',on=['query_id','product_id'])


# In[8]:


# # lcy 0611


pro_df = parse_beike_oof(r"../user_data/embeding_data/result_0611_20box_max/test/test.txt",'lcy061101')
testa_df = testa_df.merge(pro_df,how='left',on=['query_id','product_id'])
pro_df = parse_beike_oof(r"../user_data/embeding_data/result_0611_20box_max/val/val_10000.txt",'lcy061101')
valid_df = valid_df.merge(pro_df,how='left',on=['query_id','product_id'])

pro_df = parse_beike_oof(r"../user_data/embeding_data/result_0611_5box_aaa/test/test.txt",'lcy061102')
testa_df = testa_df.merge(pro_df,how='left',on=['query_id','product_id'])
pro_df = parse_beike_oof(r"../user_data/embeding_data/result_0611_5box_aaa/val/val_10000.txt",'lcy061102')
valid_df = valid_df.merge(pro_df,how='left',on=['query_id','product_id'])


pro_df = parse_beike_oof(r"../user_data/embeding_data/result_0611_new_hard/test/test.txt",'lcy061103')
testa_df = testa_df.merge(pro_df,how='left',on=['query_id','product_id'])
pro_df = parse_beike_oof(r"../user_data/embeding_data/result_0611_new_hard/val/val_10000.txt",'lcy061103')
valid_df = valid_df.merge(pro_df,how='left',on=['query_id','product_id'])


# In[9]:


# tmp
pro_df = parse_beike_oof(r"../user_data/embeding_data/beike/val_21.txt",'tmp')
valid_df = valid_df.merge(pro_df,how='left',on=['query_id','product_id'])
# oof_result = get_top5_ans(valid_df.reset_index(),"tmp")
# oof_result.to_csv("tmp.csv",index=None)
# print(fea)
# !python local_val.py valid_answer.json tmp.csv result.txt


# In[10]:


#hahaha

pro_df = pd.read_csv(r"../user_data/embeding_data/beike/haha/test637.csv").rename(columns={
    "query":"query_id","product":"product_id","proba":'hahaha'})
testa_df = testa_df.merge(pro_df,how='left',on=['query_id','product_id'])

pro_df = pd.read_csv(r"../user_data/embeding_data/beike/haha/valid637.csv").rename(columns={
    "query":"query_id","product":"product_id","proba":'hahaha'})
valid_df = valid_df.merge(pro_df,how='left',on=['query_id','product_id'])


# In[11]:


#beike finetune vaild

pro_df = pd.read_csv(r"../user_data/embeding_data/beike_finetune_valid/test.csv").rename(columns={
    "query_id":"query_id","product_id":"product_id","prob":'beike_finetune_vaild'})
testa_df = testa_df.merge(pro_df,how='left',on=['query_id','product_id'])

pro_df = pd.read_csv(r"../user_data/embeding_data/beike_finetune_valid/val.csv").rename(columns={
    "query_id":"query_id","product_id":"product_id","prob":'beike_finetune_vaild'})
valid_df = valid_df.merge(pro_df,how='left',on=['query_id','product_id'])


# In[12]:


#beike finetune vaild
# 0611

pro_df = pd.read_csv(r"../user_data/embeding_data/beike_finetune_valid/test(1).csv").rename(columns={
    "query_id":"query_id","product_id":"product_id","prob":'beike_finetune_vaild_0611'})
testa_df = testa_df.merge(pro_df,how='left',on=['query_id','product_id'])

pro_df = pd.read_csv(r"../user_data/embeding_data/beike_finetune_valid/val(1).csv").rename(columns={
    "query_id":"query_id","product_id":"product_id","prob":'beike_finetune_vaild_0611'})
valid_df = valid_df.merge(pro_df,how='left',on=['query_id','product_id'])


# In[13]:


#beike finetune vaild
# 0611-2

pro_df = pd.read_csv(r"../user_data/embeding_data/beike_finetune_valid/test(2).csv").rename(columns={
    "query_id":"query_id","product_id":"product_id","prob":'beike_finetune_vaild_0611_2'})
testa_df = testa_df.merge(pro_df,how='left',on=['query_id','product_id'])

pro_df = pd.read_csv(r"../user_data/embeding_data/beike_finetune_valid/val(2).csv").rename(columns={
    "query_id":"query_id","product_id":"product_id","prob":'beike_finetune_vaild_0611_2'})
valid_df = valid_df.merge(pro_df,how='left',on=['query_id','product_id'])


# In[14]:


#zsy
pro_df = pd.read_csv(r"../user_data/embeding_data/beike/add_lda_fea_oof.csv").rename(columns={"oof_pred":'lda'})

valid_df = valid_df.merge(pro_df,how='left',on=['query_id','product_id'])

pro_df = pd.read_csv(r"../user_data/embeding_data/beike/add_lda_fea_testb.csv").rename(columns={"prob":'lda'})
testa_df = testa_df.merge(pro_df,how='left',on=['query_id','product_id'])

pro_df = pd.read_csv(r"../user_data/embeding_data/beike/add_w2v_fea_oof.csv").rename(columns={"oof_pred":'w2v'})
valid_df = valid_df.merge(pro_df,how='left',on=['query_id','product_id'])

pro_df = pd.read_csv(r"../user_data/embeding_data/beike/add_w2v_fea_testb.csv").rename(columns={"prob":'w2v'})
testa_df = testa_df.merge(pro_df,how='left',on=['query_id','product_id'])


# In[15]:


#imgoof
pro_df = pd.read_csv(r"../user_data/embeding_data/zq/img_oof054valid.csv")
valid_df = valid_df.merge(pro_df,how='left',on=['query_id','product_id'])

pro_df = pd.read_csv(r"../user_data/embeding_data/zq/img_oof054test.csv")
testa_df = testa_df.merge(pro_df,how='left',on=['query_id','product_id'])


# In[16]:


#zq
pro_df = pd.read_csv(r"../user_data/embeding_data/zp_catboost/img_Cat_oofvalid.csv").rename(columns={"oof_pred":'zq_img_Cat_oof'})

valid_df = valid_df.merge(pro_df,how='left',on=['query_id','product_id'])

pro_df = pd.read_csv(r"../user_data/embeding_data/zp_catboost/img__Cat_ooftest.csv").rename(columns={"oof_pred":'zq_img_Cat_oof'})
testa_df = testa_df.merge(pro_df,how='left',on=['query_id','product_id'])
# -----------------------------------
pro_df = pd.read_csv(r"../user_data/embeding_data/zp_catboost/img_lgboofvalid.csv").rename(columns={"oof_pred":'zq_img_lgboof'})
valid_df = valid_df.merge(pro_df,how='left',on=['query_id','product_id'])

pro_df = pd.read_csv(r"../user_data/embeding_data/zp_catboost/img_lgbooftest.csv").rename(columns={"oof_pred":'zq_img_lgboof'})
testa_df = testa_df.merge(pro_df,how='left',on=['query_id','product_id'])



# 6/11/15：19
# -----------------------------------
pro_df = pd.read_csv(r"../user_data/embeding_data/zq/img_lgboof626valid.csv").rename(columns={"oof_pred":'img_lgboof626'})

valid_df = valid_df.merge(pro_df,how='left',on=['query_id','product_id'])

pro_df = pd.read_csv(r"../user_data/embeding_data/zq/img_lgboof626test.csv").rename(columns={"oof_pred":'img_lgboof626'})
testa_df = testa_df.merge(pro_df,how='left',on=['query_id','product_id'])


# In[25]:


img_oral_fea_mean = ["features_"+str(i)for i in range(2048)]
FEATURES = img_oral_fea_mean
oof_list = ["hahaha","beike1","beike2","beike3","beike4","beike5","beike6",
            "beike7","beike8","beike9","beike10","beike11","beike12","lda","w2v",
            "img_oof","lcy060901","lcy060902","lcy061001","lcy061002",
           "beike_finetune_vaild","zq_img_Cat_oof","zq_img_lgboof","img_lgboof626",
            "beike_finetune_vaild_0611","beike_finetune_vaild_0611_2",
]

oof_list = ["beike1","beike2","beike3","beike4","beike5","beike6",    # 最高得分
            "beike7","beike8","beike9","beike10","beike11","beike12",
            "lcy060901","lcy060902","lcy061001","lcy061002",
            # "img_oof","lda","w2v","hahaha"
            ]
print('*'*20)
print('finish loading')
print('use prob', oof_list)

#  0611
#  "beike_finetune_vaild"
# ,"zq_img_Cat_oof","zq_img_lgboof","img_lgboof626",
#   "beike_finetune_vaild_0611","beike_finetune_vaild_0611_2"


# ,"lcy061102","lcy061103"
# beike top3:  7 8 9
for fea in oof_list:
    oof_result = get_top5_ans(valid_df.reset_index(),fea)
    oof_result.to_csv("../user_data/tmp_data/tmp.csv",index=None)
    print(fea)
    os.system('python local_val.py ../data/valid/valid_answer.json ../user_data/tmp_data/tmp.csv ../user_data/tmp_data/result.txt')
# oof_list = ["hahaha","lda","w2v","img_oof"] + ["beike7","beike8","beike12","beike1"]


# In[27]:



FEATURES = oof_list #+ img_oral_fea_mean #+ ["hahaha"]  #+ ["img_oof"]  + ["beike1","beike2","beike3"]  + img_oral_fea_mean

# ["hahaha"]  + ["img_oof"] + img_oral_fea_mean   0.7213626058749869
# ["hahaha"]  + ["img_oof"] + img_oral_fea_mean + ["beike1","beike2","beike3"]    result: 0.7409489950243753
# ["hahaha"]  + ["img_oof"]  + ["beike1","beike2","beike3"]   修正后 ： 0.7089463017937613    0.688
# ["hahaha"]  + ["img_oof"]                                             0.6865797908802346    
# ["hahaha"]  0.6502305592229944
# ["hahaha","beike1","beike2","beike3","lda","w2v","img_oof"]       0.775


n_splits = 5
valid_df = valid_df.set_index(INDEX)
query_id_set = list(valid_df.index.values)
query_id_set = np.array(list(set([i[1]for i in query_id_set])))
kfolds = KFold(n_splits=n_splits, shuffle=False, random_state=402)


# In[28]:


params = {
    'task': 'train',  
    'boosting_type': 'gbrt',  
    'objective': 'lambdarank',  
    'metric': 'ndcg', 
    'max_position': 5,  
    'metric_freq': 5, 
    'train_metric': True,  
    'ndcg_at': [5],
    'max_bin': 255,  
#     'num_iterations': 50,  
    'learning_rate': 0.1,  
    'num_leaves': 3,  
    # 'max_depth':6,
    'tree_learner': 'serial', 
    'min_data_in_leaf': 30,  
    'verbose': 20,
    "early_stopping":200
}
test_prob = np.zeros(len(testa_df))
valid_df["oof_pred"] = np.zeros(len(valid_df))
for (trn_idx, val_idx) in kfolds.split(query_id_set):
    print("FEATURES_len:",len(FEATURES))
    train_query_index = query_id_set[trn_idx]
    val_query_index = query_id_set[val_idx]
    t_train = valid_df.loc[(slice(None),train_query_index),:].reset_index()
    oof_index = valid_df.loc[(slice(None),val_query_index),:].index
    t_val = valid_df.loc[(slice(None),val_query_index),:].reset_index()
    train_grp = t_train.groupby("query_id",as_index=False).size().values.reshape(-1,)
    val_grp = t_val.groupby("query_id",as_index=False).size().values.reshape(-1,)
    
    train_data = lgb.Dataset(t_train[FEATURES], t_train["rank_score"], group=train_grp)
    val_data = lgb.Dataset(t_val[FEATURES], t_val["rank_score"], group=val_grp)
    gbm = lgb.train(params, train_data, valid_sets=[val_data],num_boost_round=100,early_stopping_rounds=100,verbose_eval=100,)
    test_prob += gbm.predict(testa_df[FEATURES],num_iteration=gbm.best_iteration)/n_splits
    valid_df.loc[oof_index,"oof"] = gbm.predict(t_val[FEATURES],num_iteration=gbm.best_iteration)
    
oof_result = get_top5_ans(valid_df.reset_index(),"oof")    
testa_df['oof'] = test_prob
oof_result.to_csv("../user_data/tmp_data/oof.csv",index=None)
os.system('python local_val.py ../data/valid/valid_answer.json ../user_data/tmp_data/oof.csv ../user_data/tmp_data/result.txt')


# In[30]:


# pred = gbm.predict(testa_df[FEATURES])
testa_df['rank_score'] = test_prob
sub = get_top5_ans(testa_df,"rank_score")
import zipfile
import os

def zipFile(filepath,outFullName):
    if not os.path.exists(filepath):
        print('not found')
    zip = zipfile.ZipFile(outFullName,"w",zipfile.ZIP_DEFLATED)
    zip.write(filepath, filepath)
    zip.close()

sub.to_csv("../prediction_result/submission.csv",index=None)
# zipFile('../prediction_result/submission.csv','0610_max_score.zip')  # 官方貌似不要
print('finishing!!!!!!!!')
print("file location is ../prediction_result/submission.csv")
