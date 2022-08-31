import argparse
import json
import os
import random
from shutil import copy2
from time import time

import numpy as np
import sharearray
import torch
import torch.nn as nn
import tqdm
import wandb
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from lxmert.src.lxrt.optimization import BertAdam
from models.CNN import ResNet
from models.LXMERTEnsembleGuesserOnly import LXMERTEnsembleGuesserOnly
from train.SL.parser import preprocess_config
from train.SL.vis import Visualise
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
    parser.add_argument("-ckpt", type=str, help='path to stored checkpoint', default=None)
    parser.add_argument("-exp_tracker", type=str,
                        help='track experiment using various framework, currently supports W&B: use wandb',
                        default=None)

    args = parser.parse_args()
    print(args.exp_name)
    device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
    if args.exp_tracker is not None:
        wandb.init(project="lv", entity="we")

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

    if exp_config['save_models']:
        if args.ckpt is not None:
            model_dir = os.path.dirname(args.ckpt)
        else:
            model_dir = exp_config['save_models_path'] + args.bin_name + exp_config['ts'] + '/'
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)
        # Copying config file for book keeping
        copy2(args.config, model_dir)
        with open(model_dir + 'args.json', 'w') as f:
            json.dump(vars(args), f)  # converting args.namespace to dict

    float_tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    torch.manual_seed(exp_config['seed'])
    if use_cuda:
        torch.cuda.manual_seed_all(exp_config['seed'])

    # Init model
    model = LXMERTEnsembleGuesserOnly(**ensemble_args, from_scratch=args.from_scratch)
    # TODO Checkpoint loading

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

    # For Decider
    decider_cross_entropy = nn.CrossEntropyLoss()

    # For QGen.
    _cross_entropy = nn.CrossEntropyLoss(ignore_index=0)

    if args.resnet:
        # This was for the new image case, we don't use it
        # Takes too much time.
        raise RuntimeError("Dataset for ResNet flag not implemented!")
    else:
        dataset_train = N2NLXMERTDataset(split='train', **dataset_args, complete_only=True,
                                         imgid2fasterRCNNfeatures=imgid2fasterRCNNfeatures, num_turns=args.num_turns)
        dataset_val = N2NLXMERTDataset(split='val', **dataset_args, complete_only=True,
                                       imgid2fasterRCNNfeatures=imgid2fasterRCNNfeatures, num_turns=args.num_turns)

    print("Initializing the optimizer...")
    num_batches_per_epoch = len(dataset_train) // optimizer_args['batch_size']
    num_total_batches = num_batches_per_epoch * optimizer_args['no_epochs']
    print("Number of batches per epoch: {}".format(num_batches_per_epoch))
    print("Total number of batches: {}".format(num_total_batches))
    optimizer = BertAdam(model.parameters(), lr=optimizer_args['lr'], warmup=0.1, t_total=num_total_batches)

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_e = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
    else:
        start_e = 0
        loss = 0

    if exp_config['logging']:
        exp_config['model_name'] = 'ensemble'
        exp_config['model'] = str(model)
        exp_config['train_dataset_len'] = str(len(dataset_train))
        exp_config['valid_dataset_len'] = str(len(dataset_val))
        exp_config['modulo'] = True if args.modulo > 1 else False
        visualise = Visualise(**exp_config)

    for epoch in range(start_e, optimizer_args['no_epochs']):
        start = time()
        print('epoch', epoch)

        # Logging
        train_decision_loss = float_tensor()
        val_decision_loss = float_tensor()
        val_qgen_loss = float_tensor()
        train_guesser_loss = float_tensor()
        val_guesser_loss = float_tensor()
        train_total_loss = float_tensor()
        val_total_loss = float_tensor()

        training_guesser_accuracy = list()
        validation_guesser_accuracy = list()
        training_ask_accuracy = list()
        training_guess_accuracy = list()
        validation_ask_accuracy = list()
        validation_guess_accuracy = list()

        for split, dataset in zip(exp_config['splits'], [dataset_train, dataset_val]):
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=optimizer_args['batch_size'],
                shuffle=True,
                drop_last=False,
                pin_memory=use_cuda,
                num_workers=0
            )

            if split == 'train':
                model.train()
            else:
                model.eval()

            for i_batch, sample in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), ncols=100):
                if i_batch > 60 and args.breaking:
                    print('Breaking after processing 60 batch')
                    break

                if epoch == 14 and i_batch == 52:
                    aa = 0

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

                guesser_loss = to_var(torch.zeros(1))
                decider_loss = to_var(torch.zeros(1))

                decider_accuracy = 0
                ask_accuracy = 0

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

                decider_loss += ensemble_args['decider']['guess_weight'] * decider_cross_entropy(decider_out.squeeze(1),
                                                                                                 sample['decider_tgt'])
                guess_accuracy = calculate_accuracy(decider_out.squeeze(1), sample['decider_tgt'])

                guesser_loss += guesser_loss_function(guesser_out * sample['objects_mask'].float(),
                                                      sample['target_obj'])
                guesser_accuracy = calculate_accuracy(softmax(guesser_out), sample['target_obj'].reshape(-1))

                if args.no_decider:
                    loss = guesser_loss
                else:
                    loss = guesser_loss + decider_loss / batch_size

                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    optimizer.step()

                    # Logging variables
                    training_guesser_accuracy.append(guesser_accuracy)
                    training_guess_accuracy.append(guess_accuracy)
                    train_decision_loss = torch.cat([train_decision_loss, decider_loss.data / batch_size])
                    train_guesser_loss = torch.cat([train_guesser_loss, guesser_loss.data])
                    train_total_loss = torch.cat([train_total_loss, loss.data])

                    if exp_config['logging']:
                        visualise.iteration_update(
                            loss=loss.data[0],
                            qgen_loss=0,
                            guesser_loss=guesser_loss.data[0],
                            decider_loss=decider_loss.data[0] / batch_size,
                            ask_accuracy=0,
                            guess_accuracy=guess_accuracy,
                            guesser_accuracy=guesser_accuracy,
                            training=True,
                            epoch=epoch
                        )
                elif split == 'val':
                    validation_guesser_accuracy.append(guesser_accuracy)
                    validation_guess_accuracy.append(guess_accuracy)
                    val_decision_loss = torch.cat([val_decision_loss, decider_loss.data / batch_size])
                    val_guesser_loss = torch.cat([val_guesser_loss, guesser_loss.data])

                    val_total_loss = torch.cat([val_total_loss, loss.data])

                    if exp_config['logging']:
                        visualise.iteration_update(
                            loss=loss.data[0],
                            qgen_loss=0,
                            guesser_loss=guesser_loss.data[0],
                            decider_loss=decider_loss.data[0] / batch_size,
                            ask_accuracy=0,
                            guess_accuracy=guess_accuracy,
                            guesser_accuracy=guesser_accuracy,
                            training=False,
                            epoch=epoch
                        )

        if exp_config['save_models']:
            model_file = os.path.join(model_dir, ''.join(['model_ensemble_', args.bin_name, '_E_', str(epoch)]))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, model_file)

        print("Epoch %03d, Time taken %.3f, Total Training Loss %.4f, Total Validation Loss %.4f" % (
        epoch, time() - start, torch.mean(train_total_loss), torch.mean(val_total_loss)))
        print("Validation Loss:: QGen %.3f, Decider %.3f, Guesser %.3f" % (
        torch.mean(val_qgen_loss), torch.mean(val_decision_loss), torch.mean(val_guesser_loss)))
        print("Training Accuracy:: Guess  %.3f, Guesser %.3f" % (
        np.mean(training_guess_accuracy), np.mean(training_guesser_accuracy)))
        print("Validation Accuracy:: Guess  %.3f, Guesser %.3f" % (
        np.mean(validation_guess_accuracy), np.mean(validation_guesser_accuracy)))

        if args.exp_tracker is not None:
            wandb.log({'Guesser Training Loss': torch.mean(train_guesser_loss),
                       'Guesser Validation Loss': torch.mean(val_guesser_loss),
                       'Guesser Training Accuracy': torch.mean(torch.tensor(training_guesser_accuracy)),
                       'Guesser Validation Accuracy': torch.mean(torch.tensor(validation_guesser_accuracy)),
                       })

        if exp_config['save_models']:
            print("Saved model to %s" % (model_file))

        print('-----------------------------------------------------------------')
        if exp_config['logging']:
            visualise.epoch_update(
                train_loss=torch.mean(train_total_loss),
                train_qgen_loss=torch.Tensor([0]),
                train_guesser_loss=0 if (epoch % args.modulo != 0) else torch.mean(train_guesser_loss),
                train_decider_loss=torch.mean(train_decision_loss),
                train_ask_accuracy=torch.Tensor([0]),
                train_guess_accuracy=torch.Tensor([0]),
                train_guesser_accuracy=0 if (epoch % args.modulo != 0) else np.mean(training_guesser_accuracy),
                valid_loss=torch.mean(val_total_loss),
                valid_qgen_loss=torch.mean(val_qgen_loss),
                valid_guesser_loss=0 if (epoch % args.modulo != 0) else torch.mean(val_guesser_loss),
                valid_decider_loss=torch.mean(val_decision_loss),
                valid_ask_accuracy=torch.Tensor([0]),
                valid_guess_accuracy=torch.Tensor([0]),
                valid_guesser_accuracy=0 if (epoch % args.modulo != 0) else np.mean(validation_guesser_accuracy),
                epoch=epoch
            )
