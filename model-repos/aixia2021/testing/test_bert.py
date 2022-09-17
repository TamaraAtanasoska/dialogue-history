# import progressbar
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from models.BERTEnsemble import BERTEnsemble
from train.SL.parser import preprocess_config
from utils.datasets.SL.N2NBERTDataset import N2NBERTDataset
from utils.eval import calculate_accuracy
from utils.wrap_var import to_var

# TODO Make this capitalised everywhere to inform it is a global variable
use_cuda = torch.cuda.is_available()


def test_bert_model(best_ckpt, dataset_args, ensemble_args, optimizer_args, exp_config):
    multiple_gpus_available = torch.cuda.device_count() > 1
    device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
    random.seed(exp_config["seed"])
    np.random.seed(exp_config["seed"])
    torch.manual_seed(exp_config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(exp_config["seed"])

    # Init Model
    model = BERTEnsemble(**ensemble_args)

    if multiple_gpus_available:
        model = DataParallel(model)
    model.to(device)

    softmax = nn.Softmax(dim=-1)

    dataset_test = N2NBERTDataset(
        split="test",
        add_sep=False,
        complete_only=True,
        **dataset_args,
    )

    checkpoint = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    dataloader = DataLoader(
        dataset=dataset_test,
        batch_size=optimizer_args["batch_size"],
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=False,
        num_workers=0,
    )

    predictions = []
    targets = []
    with torch.no_grad():
        for i_batch, sample in tqdm.tqdm(
            enumerate(dataloader), total=len(dataloader), ncols=100
        ):

            sample["tgt_len"], ind = torch.sort(sample["tgt_len"], 0, descending=True)

            # Get batch
            for k, v in sample.items():
                if k == "tgt_len":
                    sample[k] = to_var(v)
                elif torch.is_tensor(v):
                    sample[k] = to_var(v[ind])
                elif isinstance(v, list):
                    sample[k] = [v[i] for i in ind]

            # Masking w.r.t decider_tgt
            masks = list()
            mask1 = sample["decider_tgt"].data

            if torch.sum(mask1) >= 1:
                masks.append(torch.nonzero(1 - mask1))
                masks.append(torch.nonzero(mask1))
            else:
                masks.append(torch.nonzero(1 - mask1))

            for idx, mask in enumerate(masks):
                # When all elements belongs to QGen or Guess only
                if len(mask) <= 0:
                    continue
                mask = mask.squeeze()

                if idx == 1:
                    mask = mask.reshape(-1)
                    # decision, guesser_out
                    guesser_out = model(
                        history_raw=[sample["history_raw"][i] for i in mask],
                        tgt_len=sample["tgt_len"][mask],
                        spatials=sample["spatials"][mask],
                        objects=sample["objects"][mask],
                        mask_select=idx,
                        target_cat=sample["target_cat"][mask],
                    )
                    predictions.append(softmax(guesser_out))
                    targets.append(sample["target_obj"][masks[1].squeeze()].reshape(-1))
                    # guesser_accuracy = calculate_accuracy(softmax(guesser_out), sample['target_obj'][masks[1].squeeze()].reshape(-1))

    accuracy = calculate_accuracy(torch.cat(predictions), torch.cat(targets))
    print(f"Guesser Accuracy: {accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="data", help="Data Directory")
    parser.add_argument(
        "-config",
        type=str,
        default="config/SL/config_devries_bert.json",
        help="Config file",
    )
    parser.add_argument("-exp_name", type=str, help="Experiment Name")
    parser.add_argument(
        "-bin_name", type=str, default="", help="Name of the trained model file"
    )
    parser.add_argument(
        "-my_cpu",
        action="store_true",
        help="To select number of workers for dataloader. CAUTION: If using your own system then make this True",
    )
    parser.add_argument(
        "-breaking",
        action="store_true",
        help="To Break training after 5 batch, for code testing purpose",
    )
    parser.add_argument(
        "-resnet",
        action="store_true",
        help="This flag will cause the program to use the image features from the ResNet forward pass instead of the precomputed ones.",
    )
    parser.add_argument(
        "-modulo",
        type=int,
        default=1,
        help="This flag will cause the guesser to be updated every modulo number of epochs",
    )
    parser.add_argument(
        "-no_decider",
        action="store_true",
        help="This flag will cause the decider to be turned off",
    )
    parser.add_argument("-from_scratch", type=bool, default=False)
    parser.add_argument("-num_turns", type=int, default=None)
    parser.add_argument(
        "-best_ckpt", type=str, default="", help="Name of the trained model file"
    )
    parser.add_argument(
        "-model_type", type=str, default="blind", help="blind or visual"
    )

    args = parser.parse_args()
    print(args.exp_name)

    # Load the Arguments and Hyperparamters
    ensemble_args, dataset_args, optimizer_args, exp_config = preprocess_config(args)
    test_bert_model(
        best_ckpt=args.best_ckpt,
        dataset_args=dataset_args,
        ensemble_args=ensemble_args,
        optimizer_args=optimizer_args,
        exp_config=exp_config,
    )
