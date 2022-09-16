import argparse
import json
import random

import numpy as np
import sharearray
import torch
import torch.nn as nn
import tqdm
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from models.CNN import ResNet
from models.LXMERTEnsembleGuesserOnly import LXMERTEnsembleGuesserOnly
from train.SL.parser import preprocess_config
from utils.datasets.SL.N2NLXMERTDataset import N2NLXMERTDataset
from utils.eval import calculate_accuracy
from utils.wrap_var import to_var

# TODO Make this capitalised everywhere to inform it is a global variable
use_cuda = torch.cuda.is_available()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="data", help='Data Directory')
    parser.add_argument("-config", type=str, default="config/SL/config_bert.json", help='Config file')
    parser.add_argument("-exp_name", type=str, help='Experiment Name')
    parser.add_argument("-bin_name", type=str, default='', help='Name of the trained model file')
    parser.add_argument("-my_cpu", action='store_true',
                        help='To select number of workers for dataloader. CAUTION: If using your own system then make this True')
    parser.add_argument("-breaking", action='store_true',
                        help='To Break training after 5 batch, for code testing purpose')
    parser.add_argument("-resnet", action='store_true',
                        help='This flag will cause the program to use the image features from the ResNet forward pass instead of the precomputed ones.')
    parser.add_argument("-modulo", type=int, default=1,
                        help='This flag will cause the guesser to be updated every modulo number of epochs')
    parser.add_argument("-no_decider", action='store_true', help='This flag will cause the decider to be turned off')
    parser.add_argument("-num_turns", type=int, default=None)
    parser.add_argument("--preloaded", type=bool, default=False)
    parser.add_argument("-from_scratch", type=bool, default=False)
    parser.add_argument('-best_ckpt', type=str, default='', help='Name of the trained model file')
    parser.add_argument('-model_type', type=str, default='blind', help='blind or visual')

    args = parser.parse_args()
    print(args.exp_name)
    device = torch.device('cuda:0') if use_cuda else torch.device('cpu')

    ensemble_args, dataset_args, optimizer_args, exp_config = preprocess_config(args)

    print("Loading MSCOCO bottomup index from: {}".format(dataset_args["FasterRCNN"]["mscoco_bottomup_index"]))
    with open(dataset_args["FasterRCNN"]["mscoco_bottomup_index"]) as in_file:
        mscoco_bottomup_index = json.load(in_file)
        image_id2image_pos = mscoco_bottomup_index["image_id2image_pos"]
        image_pos2image_id = mscoco_bottomup_index["image_pos2image_id"]
        img_h = mscoco_bottomup_index["img_h"]
        img_w = mscoco_bottomup_index["img_w"]

    print("Loading MSCOCO bottomup features from: {}".format(dataset_args["FasterRCNN"]["mscoco_bottomup_features"]))
    mscoco_bottomup_features = None
    if args.preloaded:
        print("Loading preloaded MS-COCO Bottom-Up features")
        mscoco_bottomup_features = sharearray.cache("mscoco_vectorized_features", lambda: None)
        mscoco_bottomup_features = np.array(mscoco_bottomup_features)
    else:
        mscoco_bottomup_features = np.load(dataset_args["FasterRCNN"]["mscoco_bottomup_features"])
        mscoco_bottomup_features = mscoco_bottomup_features.f.arr_0
    print("Loading MSCOCO bottomup boxes from: {}".format(dataset_args["FasterRCNN"]["mscoco_bottomup_boxes"]))
    mscoco_bottomup_boxes = None
    if args.preloaded:
        print("Loading preloaded MS-COCO Bottom-Up boxes")
        mscoco_bottomup_boxes = sharearray.cache("mscoco_vectorized_boxes", lambda: None)
        mscoco_bottomup_boxes = np.array(mscoco_bottomup_boxes)
    else:
        mscoco_bottomup_boxes = np.load(dataset_args["FasterRCNN"]["mscoco_bottomup_boxes"])

    imgid2fasterRCNNfeatures = {}
    for mscoco_id, mscoco_pos in image_id2image_pos.items():
        imgid2fasterRCNNfeatures[mscoco_id] = dict()
        imgid2fasterRCNNfeatures[mscoco_id]["features"] = mscoco_bottomup_features[mscoco_pos]
        imgid2fasterRCNNfeatures[mscoco_id]["boxes"] = mscoco_bottomup_boxes[mscoco_pos]
        imgid2fasterRCNNfeatures[mscoco_id]["img_h"] = img_h[mscoco_pos]
        imgid2fasterRCNNfeatures[mscoco_id]["img_w"] = img_w[mscoco_pos]

    random.seed(exp_config['seed'])
    np.random.seed(exp_config['seed'])

    float_tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    torch.manual_seed(exp_config['seed'])
    if use_cuda:
        torch.cuda.manual_seed_all(exp_config['seed'])

    # Init model
    model = LXMERTEnsembleGuesserOnly(**ensemble_args, from_scratch=args.from_scratch)

    if use_cuda:
        model.cuda()
        model = DataParallel(model)
    print(model)

    if args.resnet:
        cnn = ResNet()

        if use_cuda:
            cnn.cuda()
            cnn = DataParallel(cnn)

    softmax = nn.Softmax(dim=-1)

    # For Guesser
    guesser_loss_function = nn.CrossEntropyLoss()

    # For QGen.
    _cross_entropy = nn.CrossEntropyLoss(ignore_index=0)

    dataset_test = N2NLXMERTDataset(split='test', **dataset_args, complete_only=True,
                                     imgid2fasterRCNNfeatures=imgid2fasterRCNNfeatures, num_turns=args.num_turns)



    checkpoint = torch.load(args.best_ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    predictions = []
    targets = []

    dataloader = DataLoader(
        dataset=dataset_test,
        batch_size=optimizer_args['batch_size'],
        shuffle=True,
        drop_last=False,
        pin_memory=use_cuda,
        num_workers=0
    )
    with torch.no_grad():
        for i_batch, sample in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), ncols=100):
            if i_batch > 60 and args.breaking:
                print('Breaking after processing 60 batch')
                break

            sample['tgt_len'], ind = torch.sort(sample['tgt_len'], 0, descending=True)
            batch_size = ind.size(0)

            # Get batch
            for k, v in sample.items():
                if k == 'tgt_len':
                    sample[k] = to_var(v)
                elif torch.is_tensor(v):
                    sample[k] = to_var(v[ind])
                elif isinstance(v, list):
                    sample[k] = [v[i] for i in ind]

            if args.resnet:
                # This is done so that during backprop the gradients dont flow through the ResNet
                img_features, avg_img_features = cnn(to_var(sample['image'].data, True))
                img_features, avg_img_features = to_var(img_features.data), to_var(avg_img_features.data)
            else:
                avg_img_features = sample['image']


            decider_out, guesser_out = model(
                src_q=sample['src_q'],
                tgt_len=sample['tgt_len'],
                visual_features=avg_img_features,
                spatials=sample['spatials'],
                objects=sample['objects'],
                target_cat=sample['target_cat'],
                history_raw=sample["history_raw"],
                fasterrcnn_features=sample["FasterRCNN"]["features"],
                fasterrcnn_boxes=sample["FasterRCNN"]["boxes"],
                history=sample["history"],
                history_len=sample["history_len"]
            )


            #guesser_accuracy = calculate_accuracy(softmax(guesser_out), sample['target_obj'].reshape(-1))
            predictions.append(softmax(guesser_out))
            targets.append(sample['target_obj'].reshape(-1))

        accuracy = calculate_accuracy(torch.cat(predictions), torch.cat(targets))
        print(f'Guesser Accuracy: {accuracy}')