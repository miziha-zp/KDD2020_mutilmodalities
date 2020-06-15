import numpy as np
import pandas as pd
import glob, base64
from pathlib import Path
import zipfile
import os
import json
import lightgbm as lgb
from sklearn.model_selection import KFold
from glove import *
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

def parse_base64(df):
    def _decode_rows(row):
        row['boxes'] = np.frombuffer(base64.b64decode(row['boxes']), dtype=np.float32).reshape(row['num_boxes'], 4)
        row['features'] = np.frombuffer(base64.b64decode(row['features']), dtype=np.float32).reshape(row['num_boxes'], 2048)
        row['class_labels'] = np.frombuffer(base64.b64decode(row['class_labels']), dtype=np.int64).reshape(row['num_boxes'])
        return row
    df = df.apply(lambda x: _decode_rows(x), axis=1)
    return df


# 数据存放地址
BASE_DIR = Path('../data/')

valid_df = pd.read_csv('../data/valid/valid.tsv', sep='\t')
train_df = pd.read_csv('../data/train/train.sample.tsv', sep='\t')   #ATTENTION
testa_df = pd.read_csv('../data/testB/testB.tsv', sep='\t')


valid_df = parse_base64(valid_df)
train_df = parse_base64(train_df)
testa_df = parse_base64(testa_df)

data=pd.read_csv('../data/multimodal_labels.txt',sep='\t')

#查看各类别具体明细
multimodal_labels={}
for i in range(len(data)):
    multimodal_labels[data.loc[i,'category_id']]=data.loc[i,'category_name']

#验证集ｔａｒｇｅｔ
with open("../data/valid/valid_answer.json",'r') as f:
    val_anser = json.loads(f.readlines()[0])
with open("../valid/valid_answer.json") as f:
    gt = json.load(f)


#验证集匹配ｔａｒｇｅｔ
def get_label(x):
    query_id_ = x["query_id"]
    p_id = x["product_id"]
    if p_id in gt[str(query_id_)]:
        return 1
    else:
        return 0
valid_df["label"] = valid_df.apply(get_label,axis=1)


#特征工程
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


val_anser_df['rank_score'] = 1
valid_df = valid_df.merge(val_anser_df[['product_id','query_id','rank_score']],on=['product_id','query_id'],how='left')
valid_df['rank_score'] = valid_df['rank_score'].fillna(0.5)
valid_df['rank_score'] = valid_df['rank_score'].replace(0.5,0).astype(int)


valid_df['features'] = valid_df['features'].apply(lambda x: x.mean(axis=0))
for i in range(2048):
    valid_df[f'features_{i}'] = valid_df['features'].apply(lambda x: x[i])

testa_df['features'] = testa_df['features'].apply(lambda x: x.mean(axis=0))
for i in range(2048):
    testa_df[f'features_{i}'] = testa_df['features'].apply(lambda x: x[i])



INDEX = ['product_id','query_id']

#词频特征 CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from gensim import corpora, models, similarities
from gensim.models.doc2vec import TaggedDocument
#将文本中的词语转换为词频矩阵
vectorizer = CountVectorizer()
#计算个词语出现的次数
vectorizer.fit(list(valid_df["query"])+list(testa_df["query"]))
#获取词袋中所有文本关键词
word = vectorizer.get_feature_names()
# print(word)
print("len_word",len(word))
print("len_query_id",len(set(valid_df["query"])))

wf_columns = ["wordfre"+str(index) for index,word_t in enumerate(word)]
val_X_wf = vectorizer.transform(valid_df["query"])
val_df_X_wf = pd.DataFrame(data=val_X_wf.toarray(),columns=wf_columns)
valid_df = pd.concat([valid_df,val_df_X_wf],axis=1)

testa_X_wf = vectorizer.transform(testa_df["query"])
testa_df_X_wf = pd.DataFrame(data=testa_X_wf.toarray(),columns=wf_columns)
testa_df = pd.concat([testa_df,testa_df_X_wf],axis=1)

# TfidfTransformer
vectorizer = TfidfTransformer()

val_X_wf = vectorizer.fit_transform(val_X_wf.toarray())
tf_idf_columns = ["tf-idf"+str(index) for index,word_t in enumerate(word)]
val_df_X_wf = pd.DataFrame(data=val_X_wf.toarray(),columns=tf_idf_columns)
valid_df = pd.concat([valid_df,val_df_X_wf],axis=1)

testa_X_wf = vectorizer.fit_transform(testa_X_wf.toarray())
testa_df_X_wf = pd.DataFrame(data=testa_X_wf.toarray(),columns=tf_idf_columns)
testa_df = pd.concat([testa_df,testa_df_X_wf],axis=1)


#统计特征
def get_box_stats(x):
    ans = np.zeros(33)
    for i in x:
        ans[int(i)] += 1
    return ans
valid_df["box_stat_numpy"] = valid_df["class_labels"].apply(get_box_stats)
testa_df["box_stat_numpy"] = testa_df["class_labels"].apply(get_box_stats)

# 计算当前商品所标记出来的ｂｏｘ的每个ｃｌａｓｓ的数量
box_stat_single_columns = ["box_stat_single_"+str(i)for i in range(33)]
for i in range(33):
    valid_df["box_stat_single_"+str(i)] = valid_df["box_stat_numpy"].apply(lambda x:x[i])
    testa_df["box_stat_single_"+str(i)] = testa_df["box_stat_numpy"].apply(lambda x:x[i])
    

# 计算每个搜索的ｉｄ对应的商品的ｃｌａｓｓ数量
box_stat_pool_columns = ["box_stat_pool_"+str(i)for i in range(33)]

box_stat_pool_val = valid_df.groupby(["query_id"])[box_stat_single_columns].agg("sum").rename(columns=dict(zip(box_stat_single_columns,box_stat_pool_columns)))
valid_df = valid_df.merge(box_stat_pool_val,how="left",on='query_id')

box_stat_pool_test = testa_df.groupby(["query_id"])[box_stat_single_columns].agg("sum").rename(columns=dict(zip(box_stat_single_columns,box_stat_pool_columns)))
testa_df = testa_df.merge(box_stat_pool_test,how="left",on='query_id')



# 概率特征
# 商品最大概率类别
valid_df["box_stat_single_max"] = np.argmax(valid_df[box_stat_single_columns].values,axis=1)
testa_df["box_stat_single_max"] = np.argmax(testa_df[box_stat_single_columns].values,axis=1)

# ｉｄ最大概率类别
valid_df["box_stat_pool_max"] = np.argmax(valid_df[box_stat_pool_columns].values,axis=1)
testa_df["box_stat_pool_max"] = np.argmax(testa_df[box_stat_pool_columns].values,axis=1)

# 计算两个类别是否相等
valid_df["box_stat_pool_max == box_stat_single_max"] = valid_df["box_stat_pool_max"] == valid_df["box_stat_single_max"]
testa_df["box_stat_pool_max == box_stat_single_max"] = testa_df["box_stat_pool_max"] == testa_df["box_stat_single_max"]

box_stat_3 = ["box_stat_pool_max","box_stat_single_max","box_stat_pool_max == box_stat_single_max"]





# NLP特征
def lda_model(df,size = 16):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from gensim import corpora, models, similarities
    from gensim.models.doc2vec import TaggedDocument
    dictionary = corpora.Dictionary(df['query'].values)
    corpus = [dictionary.doc2bow(text) for text in df['query'].values]
    '''
    LDA文档主题生成模型，也称三层贝叶斯概率模型，包含词、主题和文档三层结构。

    本程序是提取出每一行语料的２０个主题词
    col　每一行代表数据中的每一行，有２０个主题词
    '''
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=size)
    col = np.zeros((df.shape[0],size))
    ans = lda.get_document_topics(corpus)
    for i in tqdm(range(df.shape[0])):
        for j in ans[i]:
            col[i][j[0]] = j[1]

    df_agg = pd.DataFrame(col)
    df_agg = df_agg.add_prefix("LDA_TOPIC_{}_".format('query'))
    return df_agg

# 读取ｔｒａｉｎ中１１０００００数据训练ｗｏｒｄ２ｖｅｃ和ＤＮＮ
train_data = pd.read_csv(BASE_DIR/'train.tsv',sep='\t',chunksize=100000)
training_data = pd.DataFrame()
fea =['product_id'] + ["features_"+str(i)for i in range(2048)] + ['query']
for index,data in enumerate(train_data):
    print("=====",index,"=====")
    data = parse_base64(data)
    data['features'] = data['features'].apply(lambda x: x.mean(axis=0))
    for i in range(2048):
        data[f'features_{i}'] = data['features'].apply(lambda x: x[i])
    data = data[fea]
    training_data = pd.concat([training_data,data])
    if index > 9:
        break

fea =['product_id'] + ["features_"+str(i)for i in range(2048)] + ['query']
pic_fea = ["features_"+str(i)for i in range(2048)]


df_lda_1 = pd.DataFrame()
df_lda_1 = training_data[fea]
df_lda_1['query']=df_lda_1['query'].apply(lambda x:x.split(' '))
lda_label = lda_model(df_lda_1,size = 16)

from keras import regularizers
from keras import models
from keras import layers
import numpy as np
import keras
# 导入数据
from keras.datasets import imdb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split

from keras.layers import *
from keras.models import *
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
from keras.callbacks import *
from keras.layers.advanced_activations import LeakyReLU, PReLU
import keras.backend as K
from keras.utils import multi_gpu_model
import keras

input1 = keras.layers.Input(shape=(2048,))
x1_1 = keras.layers.Dense(2048)(input1)
x1_1 = keras.layers.Activation('relu')(x1_1)

x1_2 = keras.layers.Dense(512)(x1_1)
x1_2 = keras.layers.Activation('relu')(x1_2)

x1_3 = keras.layers.Dense(256)(x1_2)
x1_3 = keras.layers.BatchNormalization()(x1_3)
x1_3 = keras.layers.Activation('relu')(x1_3)

x1_4 = keras.layers.Dense(128)(x1_3)
x1_4 = keras.layers.Activation('relu')(x1_4)

x1_5 = keras.layers.Dense(64)(x1_4)
x1_5 = keras.layers.Activation('relu')(x1_5)

x1_6 = keras.layers.Dense(32)(x1_5)
x1_6 = keras.layers.Activation('relu')(x1_6)

x1_7 = keras.layers.Dense(16)(x1_6)

model3 = keras.models.Model(inputs=input1, outputs=x1_7)

model3 = multi_gpu_model(model3, 2)
model3.compile(optimizer = 'rmsprop',loss = 'mse',metrics = ['mse'])

# 训练网络模型
batch_size = 256
epochs = 20
df_lda_1 = df_lda_1[pic_fea].values
lda_label = lda_label.values
X_train, X_test, y_train, y_test = train_test_split(df_lda_1, w2v_label, test_size=0.2)
history=model3.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_test,y_test))

#验证集添加新特征
valid_df['query'] = valid_df['query'].apply(lambda x:x.split(' '))
lda_val = lda_model(valid_df,size = 16).values
pic_val = model3.predict(valid_df[pic_fea].values)
val_fea = pd.DataFrame(pic_val-lda_val)
val_fea = val_fea.add_prefix("pic_word_")
valid_df = pd.concat([valid_df,val_fea],axis = 1)

# 测试集添加新特征
testa_df['query'] = testa_df['query'].apply(lambda x:x.split(' '))
lda_val = lda_model(testa_df,size = 16).values
pic_val = model3.predict(testa_df[pic_fea].values)
test_fea = pd.DataFrame(pic_val-lda_val)
test_fea = test_fea.add_prefix("pic_word_")
testa_df = pd.concat([testa_df,test_fea],axis = 1)


# 训练模型
LABEL = ['rank_score','label']
NOT_FEA_COLS = INDEX + LABEL + ['boxes','features','class_labels','query','oof_pred']
img_fea_raw = ['image_h', 'image_w', 'num_boxes'] 
img_fea = ["features_"+str(i) for i in range(2048)]
NN_fea = ['pic_word_'+str(i) for i in range(16)]
FEATURES = img_fea + img_fea_raw  + list(wf_columns) +list(tf_idf_columns) +\
            box_stat_single_columns + box_stat_pool_columns+box_stat_3+NN_fea


params = {
    'task': 'train',  
    'boosting_type': 'gbdt',  
    'objective': 'lambdarank',  
    'metric': 'ndcg', 
    'max_position': 5,  
    'metric_freq': 5, 
    'train_metric': True,  
    'ndcg_at': [5],
    'max_bin': 255,  
    'num_iterations': 1000,  
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
    # 匹配所有INDEX［１］　＝＝　目标元素的数据
    t_train = valid_df.loc[(slice(None),train_query_index),:].reset_index()
    oof_index = valid_df.loc[(slice(None),val_query_index),:].index
    t_val = valid_df.loc[(slice(None),val_query_index),:].reset_index()
    train_grp = t_train.groupby("query_id",as_index=False).size().values.reshape(-1,)
    val_grp = t_val.groupby("query_id",as_index=False).size().values.reshape(-1,)
    
    train_data = lgb.Dataset(t_train[FEATURES], t_train["rank_score"], group=train_grp)
    val_data = lgb.Dataset(t_val[FEATURES], t_val["rank_score"], group=val_grp)
    gbm = lgb.train(params, train_data, valid_sets=[val_data],num_boost_round=10000,early_stopping_rounds=100)
    test_prob += gbm.predict(testa_df[FEATURES])/n_splits
    valid_df.loc[oof_index,"oof_pred"] = gbm.predict(t_val[FEATURES])
    
oof_result = get_top5_ans(valid_df.reset_index(),"oof_pred")

# 保存
valid_df.reset_index()
sub_oof_lda = valid_df['query_id','product_id','oof_pred']
sub_oof_lda.to_csv('../user_data/tmp_data/add_lda_fea_val.csv',index = None)


testa_df['prob'] = test_prob
sub_lda = testa_df[['query_id','product_id','prob']]
sub_lda.to_csv('../user_data/tmp_data/add_lda_fea_testb.csv',index = None)