#!/usr/bin/env python
# coding: utf-8

# imgoral + hahaha 0.70622908099897

# In[6]:


import numpy as np
import pandas as pd
import swifter
import glob, base64
from pathlib import Path
import zipfile
import os
from tqdm import tqdm_notebook as tqm
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


import json

import lightgbm as lgb
from sklearn.model_selection import KFold


# In[22]:


valid_df = pd.read_csv('../data/valid/valid.tsv', sep='\t')[:200]

testa_df = pd.read_csv('../data/testB/testB.tsv', sep='\t')[:200]


# In[23]:



# BASE_DIR = Path('../data/')

# valid_df = pd.read_csv(BASE_DIR/'valid.tsv', sep='\t')
# # train_df = pd.read_csv(BASE_DIR/'train.sample.tsv', sep='\t')
# testa_df = pd.read_csv(BASE_DIR/'testb.tsv', sep='\t')

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
with open("../data/valid/valid_answer.json",'r') as f:
    val_anser = json.loads(f.readlines()[0])
    
with open("../data/valid/valid_answer.json") as f:
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
for i in tqm(range(2048)):
    valid_df[f'features_{i}'] = valid_df['features'].apply(lambda x: x[i])

testa_df['features'] = testa_df['features'].swifter.apply(lambda x: x.mean(axis=0))
for i in tqm(range(2048)):
    testa_df[f'features_{i}'] = testa_df['features'].apply(lambda x: x[i])

INDEX = ['product_id','query_id']


# In[32]:


#img lgb oof 
img_oral_fea_mean = ["features_"+str(i)for i in range(2048)]
FEATURES = img_oral_fea_mean
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
    'learning_rate': 0.5,  
    'num_leaves': 300,  
    # 'max_depth':6,
    'tree_learner': 'serial', 
    'min_data_in_leaf': 30,  
    'verbose': 20,
    "early_stopping":200
}
n_splits = 5
valid_df = valid_df.set_index(INDEX)
query_id_set = list(valid_df.index.values)
query_id_set = np.array(list(set([i[1]for i in query_id_set])))
kfolds = KFold(n_splits=n_splits, shuffle=False, random_state=402)

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
    gbm = lgb.train(params, train_data, valid_sets=[val_data],num_boost_round=200,early_stopping_rounds=100,verbose_eval=100,)
    test_prob += gbm.predict(testa_df[FEATURES])/n_splits
    valid_df.loc[oof_index,"img_oof"] = gbm.predict(t_val[FEATURES])

oof_result = get_top5_ans(valid_df.reset_index(),"img_oof")    
testa_df['img_oof'] = test_prob
oof_result.to_csv("result/oof.csv",index=None)
# get_ipython().system('python local_val.py valid_answer.json result/oof.csv output/result.txt')
os.system('local_val.py ../data/valid/valid_answer.json ../user_data/tmp_data/oof.csv ../user_data/tmp_data/result.txt')

testa_df['img_oof'].head()
valid_df = valid_df.reset_index(inplace=True)

img_oof_valid = valid_df.reset_index()[["query_id","product_id","img_oof"]]
img_oof_valid.to_csv("../user_data/tmp_data/img_oof054valid.csv",index=None)
img_oof_valid.head()

img_oof_test = testa_df.reset_index()[["query_id","product_id","img_oof"]]
img_oof_test.to_csv("../user_data/tmp_data/img_oof054test.csv",index=None)

print('file saved')
print('location is ../user_data/tmp_data/')
print('if you need use this file')
print('you need to relocation these file')
