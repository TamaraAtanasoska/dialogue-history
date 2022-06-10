# coding=utf-8
# Copyleft 2019 project LXRT.

import collections
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from lxrt.tokenization import BertTokenizer
from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.vqa_data import VQADataset, VQATorchDataset, VQAEvaluator
from tasks.vqa_model import VQAModel
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = VQADataset(splits)
    tset = VQATorchDataset(dset)
    evaluator = VQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class VQA:
    def __init__(self):
        # Datasets
        self.train_tuple = get_data_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                args.valid, bs=1024,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None
        
        # Model
        self.model = VQAModel(self.train_tuple.dataset.num_answers)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)
        
        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
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

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):

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

                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple

        question_id2img_id = {x["question_id"]: x["img_id"] for x in dset.data}
        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )
        plt.rcParams['figure.figsize'] = (12, 10)
        num_regions = 36

        count = 0

        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # Avoid seeing ground truth
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)

                for layer in [0, 4]:
                    for head in [0, 1]:
                        for datapoint in range(len(sent)):
                            print(count, len(sent))
                            count += 1
                            lang2vis_attention_probs = self.model.lxrt_encoder.model.bert.encoder.x_layers[
                                layer].lang_att_map[datapoint][head].detach().cpu().numpy()

                            vis2lang_attention_probs = self.model.lxrt_encoder.model.bert.encoder.x_layers[
                                layer].visn_att_map[datapoint][head].detach().cpu().numpy()

                            plt.clf()

                            plt.subplot(2, 3, 1)
                            plt.gca().set_axis_off()
                            plt.title("Image (regions 0-7)")
                            im = cv2.imread(os.path.join("/mnt/8tera/claudio.greco/mscoco_trainval_2014",
                                                         question_id2img_id[ques_id[datapoint].item()]) + ".jpg")
                            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                            plt.imshow(im)

                            plt.subplot(2, 3, 2)
                            plt.gca().set_axis_off()
                            plt.title("Image (regions 8-15)")
                            im = cv2.imread(os.path.join("/mnt/8tera/claudio.greco/mscoco_trainval_2014",
                                                         question_id2img_id[ques_id[datapoint].item()]) + ".jpg")
                            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                            plt.imshow(im)

                            plt.subplot(2, 3, 3)
                            plt.gca().set_axis_off()
                            plt.title("Image (regions 16-35)")
                            im = cv2.imread(os.path.join("/mnt/8tera/claudio.greco/mscoco_trainval_2014",
                                                         question_id2img_id[ques_id[datapoint].item()]) + ".jpg")
                            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                            plt.imshow(im)

                            img_info = loader.dataset.imgid2img[question_id2img_id[ques_id[datapoint].item()]]
                            img_h, img_w = img_info['img_h'], img_info['img_w']
                            unnormalized_boxes = boxes[datapoint].clone()
                            unnormalized_boxes[:, (0, 2)] *= img_w
                            unnormalized_boxes[:, (1, 3)] *= img_h

                            for i, bbox in enumerate(unnormalized_boxes):
                                if i < 8:
                                    plt.subplot(2, 3, 1)
                                elif i < 16:
                                    plt.subplot(2, 3, 2)
                                else:
                                    plt.subplot(2, 3, 3)

                                bbox = [bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()]

                                if bbox[0] == 0:
                                    bbox[0] = 2
                                if bbox[1] == 0:
                                    bbox[1] = 2

                                plt.gca().add_patch(
                                    plt.Rectangle((bbox[0], bbox[1]),
                                                  bbox[2] - bbox[0] - 4,
                                                  bbox[3] - bbox[1] - 4, fill=False,
                                                  edgecolor='red', linewidth=1)
                                )

                                plt.gca().text(bbox[0], bbox[1] - 2,
                                               '%s' % i,
                                               bbox=dict(facecolor='blue'),
                                               fontsize=9, color='white')

                            ax = plt.subplot(2, 1, 2)
                            plt.title("Cross-modal attention lang2vis")

                            tokenized_question = tokenizer.tokenize(sent[datapoint])
                            tokenized_question = ["<CLS>"] + tokenized_question + ["<SEP>"]

                            transposed_attention_map = lang2vis_attention_probs[:len(tokenized_question), :num_regions]
                            im = plt.imshow(transposed_attention_map, vmin=0, vmax=1)

                            for i in range(len(tokenized_question)):
                                for j in range(num_regions):
                                    att_value = round(transposed_attention_map[i, j], 1)
                                    text = ax.text(j, i, att_value,
                                                   ha="center", va="center", color="w" if att_value <= 0.5 else "b",
                                                   fontsize=6)

                            ax.set_xticks(np.arange(num_regions))
                            ax.set_xticklabels(list(range(num_regions)))

                            ax.set_yticks(np.arange(len(tokenized_question)))
                            ax.set_yticklabels(tokenized_question)

                            plt.tight_layout()
                            # plt.gca().set_axis_off()
                            plt.savefig(
                                "/mnt/8tera/claudio.greco/guesswhat_lxmert/guesswhat/visualization_vqa/lang2vis_question_{}_layer_{}_head_{}.png"
                                    .format(ques_id[datapoint].item(), layer, head),
                                bbox_inches='tight', pad_inches=0.5)

                            plt.close()

                            ## vis2lang

                            plt.clf()

                            plt.subplot(2, 3, 1)
                            plt.gca().set_axis_off()
                            plt.title("Image (regions 0-7)")
                            im = cv2.imread(os.path.join("/mnt/8tera/claudio.greco/mscoco_trainval_2014",
                                                         question_id2img_id[ques_id[datapoint].item()]) + ".jpg")
                            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                            plt.imshow(im)

                            plt.subplot(2, 3, 2)
                            plt.gca().set_axis_off()
                            plt.title("Image (regions 8-15)")
                            im = cv2.imread(os.path.join("/mnt/8tera/claudio.greco/mscoco_trainval_2014",
                                                         question_id2img_id[ques_id[datapoint].item()]) + ".jpg")
                            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                            plt.imshow(im)

                            plt.subplot(2, 3, 3)
                            plt.gca().set_axis_off()
                            plt.title("Image (regions 16-35)")
                            im = cv2.imread(os.path.join("/mnt/8tera/claudio.greco/mscoco_trainval_2014",
                                                         question_id2img_id[ques_id[datapoint].item()]) + ".jpg")
                            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                            plt.imshow(im)

                            img_info = loader.dataset.imgid2img[question_id2img_id[ques_id[datapoint].item()]]
                            img_h, img_w = img_info['img_h'], img_info['img_w']
                            unnormalized_boxes = boxes[datapoint].clone()
                            unnormalized_boxes[:, (0, 2)] *= img_w
                            unnormalized_boxes[:, (1, 3)] *= img_h

                            for i, bbox in enumerate(unnormalized_boxes):
                                if i < 8:
                                    plt.subplot(2, 3, 1)
                                elif i < 16:
                                    plt.subplot(2, 3, 2)
                                else:
                                    plt.subplot(2, 3, 3)

                                bbox = [bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()]

                                if bbox[0] == 0:
                                    bbox[0] = 2
                                if bbox[1] == 0:
                                    bbox[1] = 2

                                plt.gca().add_patch(
                                    plt.Rectangle((bbox[0], bbox[1]),
                                                  bbox[2] - bbox[0] - 4,
                                                  bbox[3] - bbox[1] - 4, fill=False,
                                                  edgecolor='red', linewidth=1)
                                )

                                plt.gca().text(bbox[0], bbox[1] - 2,
                                               '%s' % i,
                                               bbox=dict(facecolor='blue'),
                                               fontsize=9, color='white')

                            ax = plt.subplot(2, 1, 2)
                            plt.title("Cross-modal attention vis2lang")

                            tokenized_question = tokenizer.tokenize(sent[datapoint])
                            tokenized_question = ["<CLS>"] + tokenized_question + ["<SEP>"]

                            transposed_attention_map = vis2lang_attention_probs.transpose()[:len(tokenized_question), :num_regions]
                            im = plt.imshow(transposed_attention_map, vmin=0, vmax=1)

                            for i in range(len(tokenized_question)):
                                for j in range(num_regions):
                                    att_value = round(transposed_attention_map[i, j], 1)
                                    text = ax.text(j, i, att_value,
                                                   ha="center", va="center", color="w" if att_value <= 0.5 else "b",
                                                   fontsize=6)

                            ax.set_xticks(np.arange(num_regions))
                            ax.set_xticklabels(list(range(num_regions)))

                            ax.set_yticks(np.arange(len(tokenized_question)))
                            ax.set_yticklabels(tokenized_question)

                            plt.tight_layout()
                            # plt.gca().set_axis_off()
                            plt.savefig(
                                "/mnt/8tera/claudio.greco/guesswhat_lxmert/guesswhat/visualization_vqa/vis2lang_question_{}_layer_{}_head_{}.png"
                                    .format(ques_id[datapoint].item(), layer, head),
                                bbox_inches='tight', pad_inches=0.5)

                            plt.close()


                            # print(datapoint, len(sent))
                    #
                    #         print(datapoint)
                    #         if datapoint > 20:
                    #             break
                    #     if datapoint > 20:
                    #         break
                    # if datapoint > 20:
                    #     break

                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(quesid2ans)

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

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

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

    # Test or Train
    if args.test is not None:
        # args.fast = args.tiny = False       # Always loading all data in test
        if 'test' in args.test:
            vqa.predict(
                get_data_tuple(args.test, bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'test_predict.json')
            )

        elif 'val' in args.test:    
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            result = vqa.evaluate(
                get_data_tuple('minival', bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'minival_predict.json')
            )
            print(result)
        else:
            assert False, "No such test option for %s" % args.test
    else:
        print('Splits in Train data:', vqa.train_tuple.dataset.splits)
        if vqa.valid_tuple is not None:
            print('Splits in Valid data:', vqa.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (vqa.oracle_score(vqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        vqa.train(vqa.train_tuple, vqa.valid_tuple)


