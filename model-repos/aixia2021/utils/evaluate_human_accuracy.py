import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.Ensemble import Ensemble
from train.SL.parser import preprocess_config
from utils.datasets.SL.N2NDataset import N2NDataset
from utils.eval import calculate_accuracy
from utils.model_loading import load_model
# TODO Make this capitalised everywhere to inform it is a global variable
from utils.model_utils import get_number_parameters

use_cuda = torch.cuda.is_available()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="data", help='Data Directory')
    parser.add_argument("-config", type=str, default="config/SL/config_bert.json")
    parser.add_argument("-my_cpu", action='store_true')
    parser.add_argument("-breaking", action='store_true')
    parser.add_argument("-exp_name", type=str)
    parser.add_argument("-bin_name", type=str)
    parser.add_argument("-num_regions", type=int)
    parser.add_argument("-load_bin_path", type=str)
    args = parser.parse_args()
    print(args.exp_name)

    ensemble_args, dataset_args, optimizer_args, exp_config = preprocess_config(args)

    float_tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    torch.manual_seed(exp_config['seed'])
    if use_cuda:
        torch.cuda.manual_seed_all(exp_config['seed'])

    model = Ensemble(**ensemble_args)
    print("Loading model: {}".format(args.load_bin_path))
    model = load_model(model, args.load_bin_path, use_dataparallel=use_cuda)
    model.eval()

    print("Number of parameters: {}".format(get_number_parameters(model)))

    if use_cuda:
        model.cuda()
        # model = DataParallel(model)
    print(model)

    for num_turn in [5]:
        print("NUM TURN: {}".format(num_turn))

        final_accuracies = []

        dataset_test = N2NDataset(split='val', **dataset_args, complete_only=True)
        print("The dataset contains {} instances".format(len(dataset_test)))

        dataloader = DataLoader(
            dataset=dataset_test,
            batch_size=32,
            shuffle=False,
            drop_last=False,
            pin_memory=use_cuda,
            num_workers=0
        )

        with open(os.path.join(args.data_dir, dataset_args['data_paths']['vocab_file'])) as file:
            vocab = json.load(file)
        word2i = vocab['word2i']
        i2word = vocab['i2word']

        softmax = nn.Softmax(dim=-1)

        accuracies_gt = []
        accuracies_shuffled = []
        accuracies_reversed = []

        for i_batch, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
            if i_batch > 100 and args.breaking:
                print('Breaking after processing 60 batch')
                break

            _, guesser_out = model(
                src_q=sample['src_q'],
                tgt_len=sample['tgt_len'],
                visual_features=sample["image"],
                spatials=sample['spatials'],
                objects=sample['objects'],
                mask_select=1,
                target_cat=sample['target_cat'],
                history=sample["history"],
                history_len=sample['history_len']
            )

            accuracies_gt.append(calculate_accuracy(softmax(guesser_out), sample['target_obj'].reshape(-1).cuda()))

            new_history = []
            for h in sample["history"]:
                new_h = []
                history_turns = []
                new_turn = False
                indexes = h[1:]
                for token_index, token in enumerate(indexes):
                    new_h.append(token.item())
                    if new_turn:
                        history_turns.append(new_h)
                        new_h = []
                        new_turn = False
                    if token == word2i["?"] and indexes[token_index+1] in [5, 6, 7]:
                        new_turn = True

                new_history.append(history_turns)

            shuffled_history = [x[:-1] for x in new_history]
            # shuffled_history = [random.sample(x, len(x)) for x in new_history]

            new_shuffled_history = []
            for history in shuffled_history:
                utterance = [1]
                for turn in history:
                    for token in turn:
                        utterance.append(token)
                utterance.extend([word2i['<padding>']] * (sample["history"].shape[1] - len(utterance)))
                new_shuffled_history.append(utterance)
            new_shuffled_history = torch.LongTensor(new_shuffled_history)

            _, guesser_out = model(
                src_q=sample['src_q'],
                tgt_len=sample['tgt_len'],
                visual_features=sample["image"],
                spatials=sample['spatials'],
                objects=sample['objects'],
                mask_select=1,
                target_cat=sample['target_cat'],
                history=new_shuffled_history,
                history_len=sample['history_len']
            )

            accuracies_shuffled.append(calculate_accuracy(softmax(guesser_out), sample['target_obj'].reshape(-1).cuda()))

            # reversed_history = [list(reversed(x)) for x in new_history]
            #
            # new_reversed_history = []
            # for history in reversed_history:
            #     utterance = [1]
            #     for turn in history:
            #         for token in turn:
            #             utterance.append(token)
            #     utterance.extend([word2i['<padding>']] * (sample["history"].shape[1] - len(utterance)))
            #     new_reversed_history.append(utterance)
            # new_reversed_history = torch.LongTensor(new_reversed_history)
            #
            # _, guesser_out = model(
            #     src_q=sample['src_q'],
            #     tgt_len=sample['tgt_len'],
            #     visual_features=sample["image"],
            #     spatials=sample['spatials'],
            #     objects=sample['objects'],
            #     mask_select=1,
            #     target_cat=sample['target_cat'],
            #     history=new_reversed_history,
            #     history_len=sample['history_len']
            # )
            #
            # accuracies_reversed.append(calculate_accuracy(softmax(guesser_out), sample['target_obj'].reshape(-1).cuda()))

        print("Accuracy GT: {}".format(np.mean(accuracies_gt)))
        print("Accuracy Shuffled: {}".format(np.mean(accuracies_shuffled)))
        # print("Accuracy Reversed: {}".format(np.mean(accuracies_reversed)))

        # final_accuracies.append(np.mean(accuracies_gt))
        #
        # for v in final_accuracies:
        #     print(round(v * 100, 1))
