import argparse
import gzip
import json
import os

import numpy as np
import torch
from models.LXMERTEnsemble import LXMERTEnsemble
from torch.utils.data import DataLoader

from lxmert.src.lxrt.tokenization import BertTokenizer
from lxmert.src.utils import load_obj_tsv
from train.SL.parser import preprocess_config
from utils.datasets.SL.N2NLXMERTDataset import N2NLXMERTDataset
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

    game2bboxes = {}
    with gzip.open(os.path.join(args.data_dir, "guesswhat.valid.jsonl.gz")) as file:
        for json_game in file:
            game = json.loads(json_game.decode('utf-8'))

            if not game['status'] == 'success':
                continue

            game2bboxes[game["id"]] = game

    fasterRCNN_features = []

    fasterRCNN_train_path = dataset_args["FasterRCNN"]["train"]
    print("\nLoading FasterRCNN features from: {}".format(fasterRCNN_train_path))
    fasterRCNN_features.extend(load_obj_tsv(fasterRCNN_train_path, topk=1000 if args.breaking else None))

    fasterRCNN_val_path = dataset_args["FasterRCNN"]["val"]
    print("\nLoading FasterRCNN features from: {}".format(fasterRCNN_val_path))
    fasterRCNN_features.extend(load_obj_tsv(fasterRCNN_val_path, topk=1000 if args.breaking else None))

    imgid2fasterRCNNfeatures = {}
    for img_datum in fasterRCNN_features:
        imgid2fasterRCNNfeatures[img_datum['img_id']] = img_datum

    model = LXMERTEnsemble(**ensemble_args)
    print("Loading model: {}".format(args.load_bin_path))
    model = load_model(model, args.load_bin_path, use_dataparallel=use_cuda)
    model.eval()

    if use_cuda:
        model.cuda()
        # model = DataParallel(model)
    print(model)

    dataset_val = N2NLXMERTDataset(split='val', **dataset_args, num_turns=1, imgid2fasterRCNNfeatures=imgid2fasterRCNNfeatures)

    dataloader = DataLoader(
        dataset=dataset_val,
        batch_size=8,
        shuffle=False,
        drop_last=False,
        pin_memory=use_cuda,
        num_workers=0
    )

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased",
        do_lower_case=True
    )

    lengths = []
    for sample in dataloader:
        dialogs_first_turn = sample["history_raw"]
        for dialog_first_turn in dialogs_first_turn:
            tokenized_dialog_first_turn = tokenizer.tokenize(dialog_first_turn)
            tokenized_dialog_first_turn = ["<CLS>"] + tokenized_dialog_first_turn + ["<SEP>"]
            lengths.append(len(tokenized_dialog_first_turn))

    print("Mean length: {}".format(np.mean(lengths)))
