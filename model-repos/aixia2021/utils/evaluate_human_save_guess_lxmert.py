import argparse
import collections
import json

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

    if use_cuda:
        model.cuda()
        # model = DataParallel(model)
    print(model)

    dataset_test = N2NLXMERTDataset(split='test', **dataset_args, imgid2fasterRCNNfeatures=imgid2fasterRCNNfeatures)
    print("The dataset contains {} instances".format(len(dataset_test)))

    dataloader = DataLoader(
        dataset=dataset_test,
        batch_size=16,
        shuffle=False,
        drop_last=False,
        pin_memory=use_cuda,
        num_workers=0
    )

    softmax = nn.Softmax(dim=-1)

    save_game_accuracy = {}

    save_guess_probs = collections.defaultdict(lambda: collections.defaultdict(dict))

    ind2word = {
        5 : 'yes',
        6 : 'no',
        7 : 'n/a'
    }

    all_accuracies = []
    with torch.no_grad():
        for i_batch, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
            # if i_batch > 150: # and args.breaking:
            #     print('Breaking after processing 60 batch')
            #     break

            _, guesser_out = model(
                src_q=sample['src_q'],
                tgt_len=sample['tgt_len'],
                visual_features=sample["image"],
                spatials=sample['spatials'],
                objects=sample['objects'],
                mask_select=1,
                target_cat=sample['target_cat'],
                history_raw=sample["history_raw"],
                fasterrcnn_features=sample["FasterRCNN"]["features"],
                fasterrcnn_boxes=sample["FasterRCNN"]["boxes"]
            )

            batch_accuracies = calculate_accuracy_all(softmax(guesser_out), sample['target_obj'].reshape(-1).cuda())
            all_accuracies.append(batch_accuracies)

            g_output = softmax(guesser_out) * sample['objects_mask'].float().cuda()
            target = sample['target_obj']
            for index, g in enumerate(batch_accuracies):
                save_game_accuracy[sample['game_id'][index]] = g
                num_turns = 0
                for idd, el in enumerate(sample['history'][index].data):
                    if el == 12 and int(sample['history'][index][idd+1]) in [5, 6, 7]:
                        num_turns += 1
                if int((sample['history'][index] != 0).sum()) == 1:
                    save_guess_probs[sample['game_id'][index]][num_turns]['ans'] = "X"
                else:
                    save_guess_probs[sample['game_id'][index]][num_turns]['ans'] = ind2word[int(sample['history'][index][sample['history_len'][index] - 1])]

                save_guess_probs[sample['game_id'][index]][num_turns]['probs'] = g_output[index].data.tolist()
                save_guess_probs[sample['game_id'][index]][num_turns]['target'] = int(target[index])
                save_guess_probs[sample['game_id'][index]][num_turns]['obj_mask'] = torch.nonzero(sample['objects_mask'][index]).size(0)

    with open("guesser_only_scratch_save_guess_probs.json", mode="w") as out_file:
        json.dump(save_guess_probs, out_file)

    # print(" %.4f" % (np.mean(all_accuracies)))
    #
    # save_target_prob = collections.defaultdict(list)
    # save_target_dist = collections.defaultdict(list)
    # for k, v in save_guess_probs.items():
    #     if len(v) <= 2:
    #     # if len(v) <= 3:
    #     # if len(v) <= 2:
    #         continue
    #     target_prob = 0
    #     prob_dist=[]
    #     for idx in range(1, len(v)):
    #     # for idx in range(1, len(v)-1):
    #     # for idx in range(len(v)-2, len(v)):
    #         if idx == 1:
    #         # if idx == len(v)-2:
    #             target_prob = v[idx]['probs'][v[idx]['target']]
    #             prob_dist = v[idx]['probs'][:v[idx]['obj_mask']]
    #         else:
    #             new_target_prob = v[idx]['probs'][v[idx]['target']]
    #             new_prob_dist = v[idx]['probs'][:v[idx]['obj_mask']]
    #             save_target_prob[v[idx]['ans']].append(new_target_prob-target_prob)
    #             save_target_dist[v[idx]['ans']].append(entropy([x.item() for x in prob_dist], [x.item() for x in new_prob_dist]))
    #             target_prob = new_target_prob
    #             prob_dist = new_prob_dist
    #
    # print("GDSE-SL")
    # for k, v in save_target_prob.items():
    #     print("Change in probability assigned to the target object after '{}' answer:\t\t{}".format(k, np.mean([x.item() for x in v])))
    # print()
    # for k, v in save_target_dist.items():
    #     print("Change in overall probability distribution (KL divergence) after '{}' answer:\t{}".format(k, np.mean([x.item() for x in v])))
