import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from models.EnsembleDeVries import EnsembleDeVries
from models.EnsembleGuesserOnly import EnsembleGuesserOnly
from train.SL.parser import preprocess_config
from utils.datasets.SL.N2NDataset import N2NDataset
from utils.eval import calculate_accuracy
from utils.wrap_var import to_var

# TODO Make this capitalised everywhere to inform it is a global variable
use_cuda = torch.cuda.is_available()

def test_model(best_ckpt, model_type, dataset_args, ensemble_args, optimizer_args, exp_config):
    device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
    multiple_gpus_available = torch.cuda.device_count() > 1
    random.seed(exp_config['seed'])
    np.random.seed(exp_config['seed'])
    torch.manual_seed(exp_config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(exp_config["seed"])

    # Init model
    if model_type == "blind":
        model = EnsembleDeVries(**ensemble_args)
    else: # model_type == "visual":
        model = EnsembleGuesserOnly(**ensemble_args)
    if multiple_gpus_available:
        model = DataParallel(model)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), optimizer_args["lr"])
    checkpoint = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    softmax = nn.Softmax(dim=-1)

    dataset_test = N2NDataset(split="test", **dataset_args, complete_only=True)

    dataloader = DataLoader(
        dataset=dataset_test,
        batch_size=optimizer_args["batch_size"],
        shuffle=True,
        drop_last=False,
        pin_memory=use_cuda,
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

            avg_img_features = sample["image"]

            guesser_out = model(
                src_q=sample["src_q"],
                tgt_len=sample["tgt_len"],
                visual_features=avg_img_features,
                spatials=sample["spatials"],
                objects=sample["objects"],
                mask_select=1,
                target_cat=sample["target_cat"],
                history=sample["history"],
                history_len=sample["history_len"],
            )
            if model_type == "blind":
                predictions.append(softmax(guesser_out))
            else:
                predictions.append(
                    softmax(guesser_out[1])
                )  # guesser_out is a tuple containing (decider_out, guesser_out) See forward method of EnsembleGuesserOnly
            targets.append(sample["target_obj"].reshape(-1))
            # guesser_accuracy = calculate_accuracy(softmax(guesser_out), sample['target_obj'].reshape(-1))

    accuracy = calculate_accuracy(torch.cat(predictions), torch.cat(targets))
    print(f"Guesser Accuracy: {accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-data_dir", type=str, default="data/test", help="Data Directory"
    )
    parser.add_argument(
        "-config", type=str, default="config/SL/config_devries.json", help="Config file"
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
        "-best_ckpt", type=str, default="", help="Name of the trained model file"
    )
    parser.add_argument(
        "-model_type", type=str, default="blind", help="blind or visual"
    )

    args = parser.parse_args()

    ensemble_args, dataset_args, optimizer_args, exp_config = preprocess_config(args)

    test_model(model_type=args.model_type, best_ckpt=args.best_ckpt, dataset_args=dataset_args, ensemble_args=ensemble_args, optimizer_args=optimizer_args, exp_config=exp_config)
