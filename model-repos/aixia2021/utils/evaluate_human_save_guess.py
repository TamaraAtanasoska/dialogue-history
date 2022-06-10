import argparse
import collections
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.EnsembleGuesserOnly import EnsembleGuesserOnly
from train.SL.parser import preprocess_config
from utils.datasets.SL.N2NDataset import N2NDataset
from utils.eval import calculate_accuracy_all
from utils.model_loading import load_model

# TODO Make this capitalised everywhere to inform it is a global variable
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

    model = EnsembleGuesserOnly(**ensemble_args)
    print("Loading model: {}".format(args.load_bin_path))
    model = load_model(model, args.load_bin_path, use_dataparallel=use_cuda)
    model.eval()

    if use_cuda:
        model.cuda()
        # model = DataParallel(model)
    print(model)

    for num_turn in [5]:
        print("NUM TURN: {}".format(num_turn))

        final_accuracies = []

        dataset_test = N2NDataset(split='test', **dataset_args)
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

        save_game_accuracy = {}

        save_guess_probs = collections.defaultdict(lambda: collections.defaultdict(dict))

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

            batch_accuracies = calculate_accuracy_all(softmax(guesser_out), sample['target_obj'].reshape(-1).cuda())

            g_output = softmax(guesser_out) * sample['objects_mask'].float().cuda()
            target = sample['target_obj']
            for index, g in enumerate(batch_accuracies):
                save_game_accuracy[sample['game_id'][index]] = g
                num_turns = 0
                for idd, el in enumerate(sample['history'][index].data):
                    if el == 12 and int(sample['history'][index][idd + 1]) in [5, 6, 7]:
                        num_turns += 1
                if int((sample['history'][index] != 0).sum()) == 1:
                    save_guess_probs[sample['game_id'][index]][num_turns]['ans'] = "X"
                else:
                    save_guess_probs[sample['game_id'][index]][num_turns]['ans'] = i2word[
                        str(int(sample['history'][index][sample['history_len'][index] - 1]))]

                save_guess_probs[sample['game_id'][index]][num_turns]['probs'] = g_output[index].data.tolist()
                save_guess_probs[sample['game_id'][index]][num_turns]['target'] = int(target[index])
                save_guess_probs[sample['game_id'][index]][num_turns]['obj_mask'] = torch.nonzero(
                    sample['objects_mask'][index]).size(0)

        with open("lstm_resnet_save_guess_probs.json", mode="w") as out_file:
            json.dump(save_guess_probs, out_file)
