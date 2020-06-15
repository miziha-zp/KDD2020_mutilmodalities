#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from collections import Counter
import numpy as np
import base64
from tqdm import tqdm
import os
import pickle
import json

mode = 'testB'
base_dir = '../../data/' + '{}/'.format(mode)
user_dir = '../../user_data/lxmert_processfile/'
save_dir = user_dir + '{}/'.format(mode)
os.makedirs(save_dir, exist_ok=True)

print(base_dir + '{}.tsv'.format(mode))
chunks = pd.read_csv(base_dir + '{}.tsv'.format(mode), sep='\t', header=0, quoting=3, chunksize=300000)
# 'product_id', 'image_h', 'image_w', 'num_boxes', 'boxes', 'features', 'class_labels', 'query', 'query_id'

counter = 0
data = []
for df in chunks:
    for entry in tqdm(df.values):
        product_id, image_h, image_w, num_boxes, boxes, features, class_labels, query, query_id = entry
        boxes = np.frombuffer(base64.b64decode(boxes), dtype=np.float32).reshape(num_boxes, 4)
        class_labels = np.frombuffer(base64.b64decode(class_labels), dtype=np.int64).reshape(num_boxes)

        data.append({
            "product_id": product_id,
            "image_h": image_h,
            "image_w": image_w,
            "num_boxes": num_boxes,
            "boxes": boxes,
            "class_labels": class_labels,
            "query": query,
            "query_id": query_id,
            "index": counter,
        })
        features = np.frombuffer(base64.b64decode(features), dtype=np.float32).reshape(num_boxes, 2048)
        np.save(save_dir + '{}.npy'.format(counter), features)
        counter += 1


pickle.dump(data, open(user_dir + '{}.pkl'.format(mode), 'wb'))


# In[4]:


import pandas as pd
from collections import Counter
import numpy as np
import base64
from tqdm import tqdm
import os
import pickle
import json

mode = 'train'
base_dir = '../../data/' + '{}/'.format(mode)
user_dir = '../../user_data/lxmert_processfile/'
save_dir = user_dir + '{}/'.format(mode)
os.makedirs(save_dir, exist_ok=True)

print(base_dir + '{}.tsv'.format(mode))
chunks = pd.read_csv(base_dir + '{}.tsv'.format(mode), sep='\t', header=0, quoting=3, chunksize=300000)
# 'product_id', 'image_h', 'image_w', 'num_boxes', 'boxes', 'features', 'class_labels', 'query', 'query_id'

counter = 0
data = []
for df in chunks:
    for entry in tqdm(df.values):
        product_id, image_h, image_w, num_boxes, boxes, features, class_labels, query, query_id = entry
        boxes = np.frombuffer(base64.b64decode(boxes), dtype=np.float32).reshape(num_boxes, 4)
        class_labels = np.frombuffer(base64.b64decode(class_labels), dtype=np.int64).reshape(num_boxes)

        data.append({
            "product_id": product_id,
            "image_h": image_h,
            "image_w": image_w,
            "num_boxes": num_boxes,
            "boxes": boxes,
            "class_labels": class_labels,
            "query": query,
            "query_id": query_id,
            "index": counter,
        })
        features = np.frombuffer(base64.b64decode(features), dtype=np.float32).reshape(num_boxes, 2048)
        np.save(save_dir + '{}.npy'.format(counter), features)
        counter += 1


pickle.dump(data, open(user_dir + '{}.pkl'.format(mode), 'wb'))


# In[5]:


import pandas as pd
from collections import Counter
import numpy as np
import base64
from tqdm import tqdm
import os
import pickle
import json

mode = 'valid'
base_dir = '../../data/' + '{}/'.format(mode)
user_dir = '../../user_data/lxmert_processfile/'
save_dir = user_dir + '{}/'.format(mode)
os.makedirs(save_dir, exist_ok=True)

print(base_dir + '{}.tsv'.format(mode))
chunks = pd.read_csv(base_dir + '{}.tsv'.format(mode), sep='\t', header=0, quoting=3, chunksize=300000)
# 'product_id', 'image_h', 'image_w', 'num_boxes', 'boxes', 'features', 'class_labels', 'query', 'query_id'

counter = 0
data = []
for df in chunks:
    for entry in tqdm(df.values):
        product_id, image_h, image_w, num_boxes, boxes, features, class_labels, query, query_id = entry
        boxes = np.frombuffer(base64.b64decode(boxes), dtype=np.float32).reshape(num_boxes, 4)
        class_labels = np.frombuffer(base64.b64decode(class_labels), dtype=np.int64).reshape(num_boxes)

        data.append({
            "product_id": product_id,
            "image_h": image_h,
            "image_w": image_w,
            "num_boxes": num_boxes,
            "boxes": boxes,
            "class_labels": class_labels,
            "query": query,
            "query_id": query_id,
            "index": counter,
        })
        features = np.frombuffer(base64.b64decode(features), dtype=np.float32).reshape(num_boxes, 2048)
        np.save(save_dir + '{}.npy'.format(counter), features)
        counter += 1


pickle.dump(data, open(user_dir + '{}.pkl'.format(mode), 'wb'))
