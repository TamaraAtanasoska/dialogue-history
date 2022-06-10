import argparse
import csv

import numpy as np
import torch
import torch.nn as nn
import tqdm
from models.LXMERTEnsemble import LXMERTEnsemble
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from lxmert.src.utils import load_obj_tsv
from train.SL.parser import preprocess_config
from utils.datasets.SL.N2NLXMERTDataset import N2NLXMERTDataset
from utils.eval import calculate_accuracy_verbose
from utils.model_loading import load_model

# TODO Make this capitalised everywhere to inform it is a global variable

use_cuda = torch.cuda.is_available()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="data", help='Data Directory')
    parser.add_argument("-config", type=str, default="config/SL/config_bert.json")
    parser.add_argument("-my_cpu", action='store_true')
    parser.add_argument("-breaking", action='store_true')
    parser.add_argument("-modulo", type=int, default=1)
    parser.add_argument("-no_decider", action='store_true')
    parser.add_argument("-exp_name", type=str)
    parser.add_argument("-bin_name", type=str)
    parser.add_argument("-load_bin_path", type=str)
    parser.add_argument("-save_tsv_path", type=str)
    args = parser.parse_args()
    print(args.exp_name)

    ensemble_args, dataset_args, optimizer_args, exp_config = preprocess_config(args)

    float_tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    torch.manual_seed(exp_config['seed'])
    if use_cuda:
        torch.cuda.manual_seed_all(exp_config['seed'])

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
    model = load_model(model, args.load_bin_path, use_dataparallel=use_cuda)
    model.eval()

    if use_cuda:
        model.cuda()
        model = DataParallel(model)
    print(model)

    dataset_val = N2NLXMERTDataset(split='val', **dataset_args, complete_only=True, imgid2fasterRCNNfeatures=imgid2fasterRCNNfeatures)

    dataloader = DataLoader(
        dataset=dataset_val,
        batch_size=8,
        shuffle=False,
        drop_last=False,
        pin_memory=use_cuda,
        num_workers=0
    )

    accuracies = []
    dialogs = []

    with open(args.save_tsv_path, mode="w") as out_file:
        writer = csv.writer(out_file, delimiter="\t")
        writer.writerow(["game_id", "image_file", "num_objects", "guesses_prob", "guesses_confidence"])

        for i_batch, sample in enumerate(tqdm.tqdm(dataloader, total=len(dataloader))):
            if i_batch > 60 and args.breaking:
                print('Breaking after processing 60 batch')
                break

            softmax = nn.Softmax(dim=-1)

            decider_out, guesser_out = model(
                src_q=sample['src_q'],
                tgt_len=sample['tgt_len'],
                visual_features=sample['image'],
                spatials=sample['spatials'],
                objects=sample['objects'],
                mask_select=1,
                target_cat=sample['target_cat'],
                history_raw=sample["history_raw"],
                fasterrcnn_features=sample["FasterRCNN"]["features"],
                fasterrcnn_boxes=sample["FasterRCNN"]["boxes"]
            )

            guesser_accuracy, guesses, guesses_probs = calculate_accuracy_verbose(softmax(guesser_out), sample['target_obj'].cuda())
            accuracies.append(guesser_accuracy)

            for i in range(sample["history"].shape[0]):
                if guesses[i] == 1:
                    masked_objects = [x for x in sample["objects"][i].cpu().tolist() if x != 0]
                    if len(masked_objects) <= 5:
                        if len(set(masked_objects)) == len(masked_objects):
                            guesses_confidence = guesses_probs[i].item() - 1 / len(set(masked_objects))
                            dialogs.append((sample["game_id"][i], sample["image_file"][i], len(masked_objects), guesses_probs[i].item(), guesses_confidence))

        dialogs = sorted(dialogs, key=lambda x: x[4], reverse=True)

        for dialog in dialogs:
            print(dialog)
            writer.writerow(dialog)

        print("Accuracy: {}".format(np.mean(accuracies)))
