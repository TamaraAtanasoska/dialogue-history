# import progressbar
import argparse
import json
import os
from shutil import copy2
from time import time

import torch
import torch.nn as nn
import tqdm
import wandb
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from lxmert.src.lxrt.optimization import BertAdam
from models.BERTEnsemble import BERTEnsemble
from models.CNN import ResNet
from train.SL.parser import preprocess_config
from utils.datasets.SL.N2NBERTDataset import N2NBERTDataset
from utils.eval import calculate_accuracy
from utils.wrap_var import to_var

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
multiple_gpus_available = torch.cuda.device_count() > 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="data", help='Data Directory')
    parser.add_argument("-config", type=str, default="config/SL/config_devries_bert.json", help='Config file')
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
    parser.add_argument("-from_scratch", type=bool, default=False)
    parser.add_argument("-num_turns", type=int, default=None)
    parser.add_argument("-ckpt", type=str, help='path to stored checkpoint', default=None)
    parser.add_argument("-exp_tracker", type=str,
                        help='track experiment using various framework, currently supports W&B: use wandb',
                        default=None)

    args = parser.parse_args()
    print(args.exp_name)
    if args.exp_tracker is not None:
        wandb.init(project="lv", entity="we")
    # Load the Arguments and Hyperparamters
    ensemble_args, dataset_args, optimizer_args, exp_config = preprocess_config(args)

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

    float_tensor = torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor
    torch.manual_seed(exp_config['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(exp_config['seed'])

    # Init Model
    model = BERTEnsemble(**ensemble_args, from_scratch=args.from_scratch)
    # TODO Checkpoint loading

    if multiple_gpus_available:
        model = DataParallel(model)
    model.to(device)
    print(model)

    if args.resnet:
        cnn = ResNet()

        if multiple_gpus_available:
            cnn = DataParallel(cnn)
        cnn.to(device)

    softmax = nn.Softmax(dim=-1)

    # For Guesser
    guesser_loss_function = nn.CrossEntropyLoss()

    # For Decider
    decider_cross_entropy = nn.CrossEntropyLoss(reduction='sum')

    # For QGen.
    _cross_entropy = nn.CrossEntropyLoss(ignore_index=0)

    dataset_train = N2NBERTDataset(split='train', add_sep=False, complete_only=True, **dataset_args,
                                   num_turns=args.num_turns)
    dataset_val = N2NBERTDataset(split='val', add_sep=False, complete_only=True, **dataset_args,
                                 num_turns=args.num_turns)

    # TODO Use different optimizers for different modules if required.
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

    for epoch in range(start_e, optimizer_args['no_epochs']):
        start = time()
        print('epoch', epoch)

        # Logging
        train_decision_loss = float_tensor()
        val_decision_loss = float_tensor()
        train_qgen_loss = float_tensor()
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
                pin_memory=True if device.type == 'cuda' else False,
                drop_last=False,
                num_workers=0
            )

            if split == 'train':
                model.train()
            else:
                model.eval()

            with torch.set_grad_enabled(split == 'train'):
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

                    # Masking w.r.t decider_tgt
                    masks = list()
                    mask1 = sample['decider_tgt'].data

                    if torch.sum(mask1) >= 1:
                        masks.append(torch.nonzero(1 - mask1))
                        masks.append(torch.nonzero(mask1))
                    else:
                        masks.append(torch.nonzero(1 - mask1))

                    word_logits_loss = to_var(torch.zeros(1))
                    guesser_loss = to_var(torch.zeros(1))
                    decider_loss = to_var(torch.zeros(1))

                    decider_accuracy = 0
                    ask_accuracy = 0
                    guess_accuracy = torch.Tensor([0])
                    guesser_accuracy = torch.Tensor([0])

                    for idx, mask in enumerate(masks):
                        # When all elements belongs to QGen or Guess only
                        if len(mask) <= 0:
                            continue
                        mask = mask.squeeze()

                        if idx == 1:
                            if epoch % args.modulo != 0:
                                continue
                            else:
                                mask = mask.reshape(-1)
                                # decision, guesser_out
                                guesser_out = model(
                                    history_raw=[sample["history_raw"][i] for i in mask],
                                    tgt_len=sample['tgt_len'][mask],
                                    spatials=sample['spatials'][mask],
                                    objects=sample['objects'][mask],
                                    mask_select=idx,
                                    target_cat=sample['target_cat'][mask]
                                )

                                guesser_loss += guesser_loss_function(
                                    guesser_out * sample['objects_mask'][mask].float(), sample['target_obj'][mask])
                                guesser_accuracy = calculate_accuracy(softmax(guesser_out),
                                                                      sample['target_obj'][masks[1].squeeze()].reshape(
                                                                          -1))

                    if args.no_decider:
                        loss = guesser_loss
                    else:
                        loss = guesser_loss + decider_loss / batch_size

                    if split == 'train':
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()

                        # Logging variables
                        training_guesser_accuracy.append(guesser_accuracy)
                        training_ask_accuracy.append(ask_accuracy)
                        training_guess_accuracy.append(guess_accuracy)
                        train_decision_loss = torch.cat([train_decision_loss, decider_loss.data / batch_size])
                        train_guesser_loss = torch.cat([train_guesser_loss, guesser_loss.data])

                        train_total_loss = torch.cat([train_total_loss, loss.data])

                    elif split == 'val':
                        validation_guesser_accuracy.append(guesser_accuracy)
                        validation_ask_accuracy.append(ask_accuracy)
                        validation_guess_accuracy.append(guess_accuracy)
                        val_decision_loss = torch.cat([val_decision_loss, decider_loss.data / batch_size])
                        val_guesser_loss = torch.cat([val_guesser_loss, guesser_loss.data])

                        val_total_loss = torch.cat([val_total_loss, loss.data])

        #  and (epoch%args.modulo == 0)
        if exp_config['save_models']:
            model_file = os.path.join(model_dir, ''.join(['model_ensemble_', args.bin_name, '_E_', str(epoch)]))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, model_file)

        if epoch % args.modulo != 0:
            print("Epoch %03d, Time taken %.3f, Total Training Loss %.4f, Total Validation Loss %.4f" % (
                epoch, time() - start, torch.mean(train_total_loss), torch.mean(val_total_loss)))
            print("Training Loss:: QGen %.3f, Decider %.3f" % (
                torch.mean(train_qgen_loss), torch.mean(train_decision_loss)))
            print("Validation Loss:: QGen %.3f, Decider %.3f" % (
                torch.mean(val_qgen_loss), torch.mean(val_decision_loss)))

        else:
            print("Epoch %03d, Time taken %.3f, Total Training Loss %.4f, Total Validation Loss %.4f"
                  % (epoch, time() - start, torch.mean(train_total_loss), torch.mean(val_total_loss)))
            print("Training Loss:: QGen %.3f, Decider %.3f, Guesser %.3f" % (
                torch.mean(train_qgen_loss), torch.mean(train_decision_loss), torch.mean(train_guesser_loss)))
            print("Validation Loss:: QGen %.3f, Decider %.3f, Guesser %.3f" % (
                torch.mean(val_qgen_loss), torch.mean(val_decision_loss), torch.mean(val_guesser_loss)))
            print('Training Accuracy:: Guess  %.3f, Guesser %.3f' % (
                torch.mean(torch.stack(training_guess_accuracy)), torch.mean(torch.tensor(training_guesser_accuracy))))
            print('Validation Accuracy:: Guess  %.3f, Guesser %.3f' % (
                torch.mean(torch.stack(validation_guess_accuracy)),
                torch.mean(torch.tensor(validation_guesser_accuracy))))

            if args.exp_tracker is not None:
                wandb.log({'Guesser Training Loss': torch.mean(train_guesser_loss),
                           'Guesser Validation Loss': torch.mean(val_guesser_loss),
                           'Guesser Training Accuracy': torch.mean(torch.tensor(training_guesser_accuracy)),
                           'Guesser Validation Accuracy': torch.mean(torch.tensor(validation_guesser_accuracy)),
                           })

        if exp_config['save_models']:
            print("Saved model to %s" % (model_file))
