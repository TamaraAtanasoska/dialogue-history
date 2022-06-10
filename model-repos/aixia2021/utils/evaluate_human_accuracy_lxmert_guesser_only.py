import argparse
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.LXMERTEnsembleGuesserOnly import LXMERTEnsembleGuesserOnly
from train.SL.parser import preprocess_config
from utils.datasets.SL.N2NLXMERTDataset import N2NLXMERTDataset
from utils.eval import calculate_accuracy_all
from utils.model_loading import load_model
# TODO Make this capitalised everywhere to inform it is a global variable
from utils.model_utils import get_number_parameters

use_cuda = torch.cuda.is_available()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="data", help='Data Directory')
    parser.add_argument("-config", type=str, default="config/SL/config_bert_scratch.json")
    parser.add_argument("-my_cpu", action='store_true')
    parser.add_argument("-breaking", action='store_true')
    parser.add_argument("-exp_name", type=str)
    parser.add_argument("-bin_name", type=str)
    parser.add_argument("-num_regions", type=int)
    parser.add_argument("-load_bin_path", type=str)
    parser.add_argument(
        "--load_mscoco_bottomup_index_json_filename",
        type=str,
        default="/mnt/8tera/claudio.greco/guesswhat_lxmert/guesswhat/lxmert/data/mscoco_imgfeat/mscoco_bottomup_index.json"
    )
    parser.add_argument(
        "--load_mscoco_bottomup_features_npy_filename",
        type=str,
        default="/mnt/8tera/claudio.greco/guesswhat_lxmert/guesswhat/lxmert/data/mscoco_imgfeat/save_mscoco_bottomup_features.npy"
    )
    parser.add_argument(
        "--load_mscoco_bottomup_boxes_npy_filename",
        type=str,
        default="/mnt/8tera/claudio.greco/guesswhat_lxmert/guesswhat/lxmert/data/mscoco_imgfeat/save_mscoco_bottomup_boxes.npy"
    )
    args = parser.parse_args()
    print(args.exp_name)

    ensemble_args, dataset_args, optimizer_args, exp_config = preprocess_config(args)

    float_tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    torch.manual_seed(exp_config['seed'])
    if use_cuda:
        torch.cuda.manual_seed_all(exp_config['seed'])

    print("Loading MSCOCO bottomup index from: {}".format(dataset_args["FasterRCNN"]["mscoco_bottomup_index"]))
    with open(dataset_args["FasterRCNN"]["mscoco_bottomup_index"]) as in_file:
        mscoco_bottomup_index = json.load(in_file)
        image_id2image_pos = mscoco_bottomup_index["image_id2image_pos"]
        image_pos2image_id = mscoco_bottomup_index["image_pos2image_id"]
        img_h = mscoco_bottomup_index["img_h"]
        img_w = mscoco_bottomup_index["img_w"]

    print("Loading MSCOCO bottomup features from: {}".format(dataset_args["FasterRCNN"]["mscoco_bottomup_features"]))
    mscoco_bottomup_features = np.load(dataset_args["FasterRCNN"]["mscoco_bottomup_features"])

    print("Loading MSCOCO bottomup boxes from: {}".format(dataset_args["FasterRCNN"]["mscoco_bottomup_boxes"]))
    mscoco_bottomup_boxes = np.load(dataset_args["FasterRCNN"]["mscoco_bottomup_boxes"])

    imgid2fasterRCNNfeatures = {}
    for mscoco_id, mscoco_pos in image_id2image_pos.items():
        imgid2fasterRCNNfeatures[mscoco_id] = dict()
        imgid2fasterRCNNfeatures[mscoco_id]["features"] = mscoco_bottomup_features[mscoco_pos]
        imgid2fasterRCNNfeatures[mscoco_id]["boxes"] = mscoco_bottomup_boxes[mscoco_pos]
        imgid2fasterRCNNfeatures[mscoco_id]["img_h"] = img_h[mscoco_pos]
        imgid2fasterRCNNfeatures[mscoco_id]["img_w"] = img_w[mscoco_pos]

    model = LXMERTEnsembleGuesserOnly(**ensemble_args)
    print("Loading model: {}".format(args.load_bin_path))
    model = load_model(model, args.load_bin_path, use_dataparallel=use_cuda)
    model.eval()

    print("Number of parameters: {}".format(get_number_parameters(model)))

    if use_cuda:
        model.cuda()
        # model = DataParallel(model)
    print(model)

    final_accuracies = []

    for num_turn in [5]:
        print("NUM TURN: {}".format(num_turn))

        dataset_val = N2NLXMERTDataset(split='test', **dataset_args, complete_only=True, imgid2fasterRCNNfeatures=imgid2fasterRCNNfeatures)
        print("The dataset contains {} instances".format(len(dataset_val)))

        dataloader = DataLoader(
            dataset=dataset_val,
            batch_size=64,
            shuffle=False,
            drop_last=False,
            pin_memory=use_cuda,
            num_workers=0
        )

        softmax = nn.Softmax(dim=-1)

        all_accuracies = defaultdict(list)
        with torch.no_grad():
            for turn_order in ["FWD"]:
                print("Order: {}".format(turn_order))

                for i_batch, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
                    if i_batch > 3 and args.breaking:
                        print('Breaking after processing 60 batch')
                        break

                    new_history_raw = []
                    for h in sample["history_raw"]:
                        turn = 0
                        new_h = ""
                        history_turns = []
                        new_turn = False
                        tokens = h.replace("?", " ?").split()
                        for token_index, token in enumerate(tokens):
                            new_h += token + " "
                            if new_turn:
                                history_turns.append(new_h.strip())
                                new_h = ""
                                new_turn = False
                            if token == "?" and tokens[token_index+1].lower() in ["yes", "no", "n/a"]:
                                new_turn = True
                                turn += 1
                        new_history_raw.append(history_turns)

                    if turn_order == "FWD":
                        new_history_raw = [" ".join(x) for x in new_history_raw]
                    elif turn_order == "BWD":
                        new_history_raw = [" ".join(list(reversed(x))) for x in new_history_raw]
                    elif turn_order == "SHUFFLE":
                        new_history_raw = [" ".join(x[:-1]) for x in new_history_raw]

                    _, guesser_out = model(
                        src_q=sample['src_q'],
                        tgt_len=sample['tgt_len'],
                        visual_features=sample["image"],
                        spatials=sample['spatials'],
                        objects=sample['objects'],
                        mask_select=1,
                        target_cat=sample['target_cat'],
                        history_raw=new_history_raw,
                        fasterrcnn_features=sample["FasterRCNN"]["features"],
                        fasterrcnn_boxes=sample["FasterRCNN"]["boxes"]
                    )

                    batch_accuracies = calculate_accuracy_all(softmax(guesser_out), sample['target_obj'].reshape(-1).cuda())
                    all_accuracies[turn_order].extend(batch_accuracies)

        for k, v in all_accuracies.items():
            print("{} -> {}".format(k, np.mean(v)))

            final_accuracies.append(np.mean(v))

    for v in final_accuracies:
        print(round(v * 100, 1))
