# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections
import random
import json

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import numpy as np


from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from vqa_model import VQAModel
from vqa_data import KDDCUPDataset, VQAEvaluator


DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    tset = KDDCUPDataset(splits)
    evaluator = VQAEvaluator(tset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=tset, loader=data_loader, evaluator=evaluator)


def accuracy(output, target):
    """Computes the accuracy for multiple binary predictions"""
    pred = torch.sigmoid(output) >= 0.5
    truth = target >= 0.5
    acc = pred.eq(truth).sum() #/ target.numel()
    return acc

class VQA:
    def __init__(self):
        # Model
        self.model = VQAModel()
        

         # Load pre-trained weights
        if args.load_lxmert is not None: 
            self.model.lxrt_encoder.load(args.load_lxmert)
            
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model)

        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
             self.model.lxrt_encoder.multi_gpu()

         # Loss and Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        
       
        self.train_tuple = get_data_tuple(
                args.train, bs=args.batch_size, shuffle=True, drop_last=True
            )
          
        self.valid_tuple = get_data_tuple(
                    args.valid, bs=1024,
                    shuffle=False, drop_last=False)
                          
        self.test_tuple = get_data_tuple(
                args.test, bs=1024, shuffle=False, drop_last=False)
                
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                      lr=args.lr,
                                      warmup=0.1,
                                      t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)

        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple, test_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        start_epoch = -1
        best_valid = 0.

        if args.RESUME:
            path_checkpoint = args.path_checkpoint   #  ../../model/checkpoint/lxr_best_%s.pth  # 断点路径
            checkpoint = torch.load(path_checkpoint)  # 加载断点

            self.model.load_state_dict(checkpoint['model_state_dict'])

            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']


        for epoch in range(start_epoch + 1 , args.epochs):
            for i, (feats, boxes, sent, _, _) in iter_wrapper(enumerate(loader)):

                # construct negative exmaples
                bs = len(sent)
                index_list = list(range(bs))

                sent_negative = []

                if args.nprob==0:
                    nprob=[0,1,2]
                else: 
                    nprob=[0,0,0,0,1]
                if random.choice(nprob):
                    for j in range(bs):
                        choice = random.choice(list(set(index_list) - {j}))
                        sent_negative.append(sent[choice])
                    sent = sent + sent_negative
                else:
                    # 自己做的负样本
                    for j in range(bs):
                        try:
                            choice = hashmap[sent[j]]
                            choice = ramdon.choice(choice)
                        except:
                            choice = sent[random.choice(list(set(index_list) - {j}))]
                        sent_negative.append(choice)
                    sent = sent + sent_negative



                sent = sent + sent_negative

                feats = torch.cat([feats, feats])
                boxes = torch.cat([boxes, boxes])

                target = torch.ones(bs, 1)
                target_negative = torch.zeros(bs, 1)
                target = torch.cat([target, target_negative])

                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logit = self.model(feats, boxes, sent)
                assert logit.dim() == target.dim() == 2
                loss = self.bce_loss(logit, target)
                loss = loss * logit.size(1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                batch_score = accuracy(logit, target)

                if i % 500 == 0:
                    print(
                        'epoch {}, Step {}/{}, Loss: {}'.format(epoch, i, len(loader), loss.item()))

            log_str = "\nEpoch %d: Loss: %0.4f Train %0.2f\n" % (epoch, loss.item(), batch_score)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple,epoch)

                self.save_checkpoint(epoch)
                self.save(epoch)
                self.test_output(test_tuple, epoch)
                print('output done!')
                if valid_score > best_valid:
                    best_valid = valid_score

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score ) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid )

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple,epoch=0):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        preds = []
        query_ids = []
        product_ids = []
        with torch.no_grad():
            for i, datum_tuple in enumerate(loader):
                feats, boxes, sent, ques_id, produce_id = datum_tuple   # Avoid seeing ground truth
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


        with open('../../user_data/lxmert_model/result/val/val_%s.txt'% epoch, 'w') as f:
            for q, p in query2product.items():
                q = str(q)
                p = str(p)
                f.write(q + ',' + p + '\n')
            f.close()

        with open('submission.csv', 'w') as f:
            f.write('query-id,product1,product2,product3,product4,product5\n')
            for q, p in query2product.items():
                p = sorted(p, key=lambda x: x[0], reverse=True)
                o = ','.join([p[i][1] for i in range(5)])
                f.write(q + ',' + o + '\n')

        os.system('python eval.py submission.csv')
        score = json.load(open('score.json'))["score"]
        return score

    def test_output(self,eval_tuple: DataTuple,epoch=0):
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
            
        with open('../../user_data/lxmert_model/result/test/test_%s.txt'% epoch, 'w') as f:
            for q, p in query2product.items():
                q = str(q) 
                p = str(p) 
                f.write(q + ',' + p + '\n')
            f.close()


    def evaluate(self, eval_tuple: DataTuple,epoch=0):
        """Evaluate all data in data_tuple."""
        score = self.predict(eval_tuple,epoch)
        return score #eval_tuple.evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid.item()] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, epoch):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % (str(epoch))))
        
    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
        }

        if not os.path.isdir('../../user_data/lxmert_model/checkpoint'):
            os.mkdir('../../user_data/lxmert_model/checkpoint')
        torch.save(checkpoint, '../../user_data/lxmert_model/checkpoint/lxr_best.pth')

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    # Build Class
    vqa = VQA()

    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        vqa.load(args.load)

    vqa.train(vqa.train_tuple, vqa.valid_tuple, vqa.test_tuple)

