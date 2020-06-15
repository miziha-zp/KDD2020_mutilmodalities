# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle

import numpy as np
from torch.utils.data import Dataset
from pandas.core.frame import DataFrame
from param import args


class KDDCUPDataset(Dataset):
    def __init__(self, name='train', data_dir=args.basedatadir,
                 debug=True, max_area_boxes=args.max_area_boxes, ROI_select = args.ROI_select):
        super().__init__()
        self.data_dir = data_dir
        self.name = name
        self.max_area_boxes = max_area_boxes
        self.ROI_select = ROI_select
        self.entries = pickle.load(open(
            os.path.join(self.data_dir, self.name + '.pkl'), 'rb'
        ))
        if debug :
            self.entries = self.entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]

        features = np.load(os.path.join(self.data_dir, self.name, str(entry['index']) + '.npy'))
        boxes = entry['boxes']

        if self.ROI_select == 0:
            features = features[:self.max_area_boxes]
            boxes = boxes[:self.max_area_boxes]
        elif self.ROI_select == 1:
            ##根据标签筛选主要的box
            data_index = []
            for index,value in enumerate(entry['class_labels']):
                if value == np.argmax(np.bincount(entry['class_labels'])):
                    data_index.append(index)

            features = features[data_index]
            boxes = boxes[data_index]


            ##根据面积来截断TOP box
            if boxes.shape[0]>self.max_area_boxes:
                area_count = []
                for index,box in enumerate(boxes):
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    area_count.append(area)

                data=DataFrame({"index":range(len(area_count)),
                                'area':area_count})

                data = data.loc[data['area'].rank()>len(area_count)-self.max_area_boxes]
                data = data.drop_duplicates(subset=['area'])
                data_index_2 = data['index'].tolist()

                features = features[data_index_2]
                boxes = boxes[data_index_2]
        
        
        features = np.pad(
            features,
            ((0, self.max_area_boxes - features.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0
        )

        
        img_h, img_w = entry['image_h'], entry['image_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h


        boxes = np.pad(
            boxes,
            ((0, self.max_area_boxes - boxes.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0
        )


        query = entry['query']
        question_id = entry['query_id']
        product_id = entry['product_id']

        return features, boxes, query, question_id, product_id



class VQAEvaluator:
    def __init__(self, dataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)


