# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections
import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np

from param import args

from vqa_model import VQAModel
from vqa_data import KDDCUPDataset, VQAEvaluator

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_data_tuple(splits: str, bs: int, shuffle=False, drop_last=False) -> DataTuple:
    tset = KDDCUPDataset(splits)
    evaluator = VQAEvaluator(tset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=tset, loader=data_loader, evaluator=evaluator)


class LXMERT:
    def __init__(self):
        # Model
        self.model = VQAModel()


        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()


        self.valid_tuple = get_data_tuple(
                    args.valid, bs=1024,
                    shuffle=False, drop_last=False)

        self.test_tuple = get_data_tuple(
            args.test, bs=1024, shuffle=False, drop_last=False)

    def predict_output(self, eval_tuple: DataTuple, mode, name):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        preds = []
        query_ids = []
        product_ids = []
        with torch.no_grad():
            for i, datum_tuple in enumerate(loader):
                feats, boxes, sent, ques_id, produce_id = datum_tuple  # Avoid seeing ground truth
                query_ids.extend(ques_id)
                product_ids.extend(produce_id)
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                logit = torch.sigmoid(logit)
                preds.append(logit.cpu().numpy())
        preds = np.concatenate(preds)

        # deal with format
        query2product = collections.defaultdict(list)
        for i, query_id in enumerate(query_ids):
            pred = preds[i]
            product_id = product_ids[i]
            query2product[str(query_id.item())].append([pred.tolist()[0], str(product_id.item())])

        print(os.getcwd())

        if mode == 0:
            print(os.getcwd())
            with open('../../user_data/lxmert_model/result/val/val_%s.txt' % name, 'w') as f:
                for q, p in query2product.items():
                    q = str(q)
                    p = str(p)
                    f.write(q + ',' + p + '\n')
                f.close()
        elif mode == 1:
            with open('../../user_data/lxmert_model/result/test/test_%s.txt' % name, 'w') as f:
                for q, p in query2product.items():
                    q = str(q)
                    p = str(p)
                    f.write(q + ',' + p + '\n')
                f.close()


    def load(self,path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    # Build Class
    lxmert = LXMERT()

    if args.load is not None:
        lxmert.load(args.load)
    #get val
    lxmert.predict_output(lxmert.valid_tuple, 0 , args.name)
    #get test
    lxmert.predict_output(lxmert.valid_tuple, 1 , args.name)

