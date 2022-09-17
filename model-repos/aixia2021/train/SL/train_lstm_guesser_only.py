import argparse
import json
import os
from shutil import copy2
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import wandb
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from models.CNN import ResNet
from models.EnsembleDeVries import EnsembleDeVries
from testing.test_lstm import test_model
from train.SL.parser import preprocess_config
from utils.datasets.SL.N2NDataset import N2NDataset
from utils.eval import calculate_accuracy
from utils.wrap_var import to_var

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
multiple_gpus_available = torch.cuda.device_count() > 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="data", help="Data Directory")
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
        "-no_decider",
        action="store_true",
        help="This flag will cause the decider to be turned off",
    )
    parser.add_argument("-num_turns", type=int, default=None)
    parser.add_argument(
        "-ckpt", type=str, help="path to stored checkpoint", default=None
    )
    parser.add_argument(
        "-exp_tracker",
        type=str,
        help="track experiment using various framework, currently supports W&B: use wandb",
        default=None,
    )
    parser.add_argument(
        "-test_data_dir", type=str, default=None, help="Test data directory"
    )

    args = parser.parse_args()
    print(args.exp_name)

    if args.exp_tracker is not None:
        wandb.init(project="lv", entity="we")

    ensemble_args, dataset_args, optimizer_args, exp_config = preprocess_config(args)

    if exp_config["save_models"]:
        if args.ckpt is not None:
            model_dir = os.path.dirname(args.ckpt)
        else:
            model_dir = (
                exp_config["save_models_path"] + args.bin_name + exp_config["ts"] + "/"
            )
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)
        # Copying config file for book keeping
        copy2(args.config, model_dir)
        with open(model_dir + "args.json", "w") as f:
            json.dump(vars(args), f)  # converting args.namespace to dict

    float_tensor = (
        torch.cuda.FloatTensor if device.type == "cuda" else torch.FloatTensor
    )
    torch.manual_seed(exp_config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(exp_config["seed"])

    # Init model
    model = EnsembleDeVries(**ensemble_args)
    if multiple_gpus_available:
        model = DataParallel(model)
    model.to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), optimizer_args["lr"])
    # TODO Checkpoint loading
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_e = checkpoint["epoch"] + 1
        loss = checkpoint["loss"]
        val_accuracies = checkpoint["val_accuracies"]
    else:
        start_e = 0
        loss = 0
        val_accuracies = []

    if args.resnet:
        cnn = ResNet()

        if multiple_gpus_available:
            cnn = DataParallel(cnn)
        cnn.to(device)

    softmax = nn.Softmax(dim=-1)

    # For Guesser
    guesser_loss_function = nn.CrossEntropyLoss()

    # For Decider
    decider_cross_entropy = nn.CrossEntropyLoss(reduction="sum")

    if args.resnet:
        # This was for the new image case, we don't use it
        # Takes too much time.
        raise RuntimeError("Dataset for ResNet flag not implemented!")
    else:
        dataset_train = N2NDataset(
            split="train", **dataset_args, complete_only=True, num_turns=args.num_turns
        )
        dataset_val = N2NDataset(
            split="val", **dataset_args, complete_only=True, num_turns=args.num_turns
        )

    for epoch in range(start_e, optimizer_args["no_epochs"]):
        start = time()
        print("epoch", epoch)

        # Logging
        train_decision_loss = float_tensor()
        val_decision_loss = float_tensor()
        train_guesser_loss = float_tensor()
        val_guesser_loss = float_tensor()
        train_total_loss = float_tensor()
        val_total_loss = float_tensor()

        training_guesser_accuracy = list()
        validation_guesser_accuracy = list()
        training_ask_accuracy = list()
        validation_ask_accuracy = list()

        for split, dataset in zip(exp_config["splits"], [dataset_train, dataset_val]):
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=optimizer_args["batch_size"],
                shuffle=True,
                drop_last=True,
                pin_memory=True if device.type == "cuda" else False,
                num_workers=0,
            )

            if split == "train":
                model.train()
            else:
                model.eval()

            for i_batch, sample in tqdm.tqdm(
                enumerate(dataloader), total=len(dataloader), ncols=100
            ):
                sample["tgt_len"], ind = torch.sort(
                    sample["tgt_len"], 0, descending=True
                )
                batch_size = ind.size(0)

                # Get batch
                for k, v in sample.items():
                    if k == "tgt_len":
                        sample[k] = to_var(v)
                    elif torch.is_tensor(v):
                        sample[k] = to_var(v[ind])
                    elif isinstance(v, list):
                        sample[k] = [v[i] for i in ind]

                if args.resnet:
                    # This is done so that during the backpropagation the gradients don't flow through the ResNet
                    img_features, avg_img_features = cnn(
                        to_var(sample["image"].data, True)
                    )
                    img_features, avg_img_features = to_var(img_features.data), to_var(
                        avg_img_features.data
                    )
                else:
                    avg_img_features = sample["image"]

                guesser_loss = to_var(torch.zeros(1))
                decider_loss = to_var(torch.zeros(1))

                decider_accuracy = 0
                ask_accuracy = 0

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

                guesser_loss += guesser_loss_function(
                    guesser_out * sample["objects_mask"].float(), sample["target_obj"]
                )
                guesser_accuracy = calculate_accuracy(
                    softmax(guesser_out), sample["target_obj"].reshape(-1)
                )

                if args.no_decider:
                    loss = guesser_loss
                else:
                    loss = guesser_loss + decider_loss / batch_size

                if split == "train":
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()

                    # Logging variables
                    training_guesser_accuracy.append(guesser_accuracy)
                    train_decision_loss = torch.cat(
                        [train_decision_loss, decider_loss.data / batch_size]
                    )
                    train_guesser_loss = torch.cat(
                        [train_guesser_loss, guesser_loss.data]
                    )
                    train_total_loss = torch.cat([train_total_loss, loss.data])

                elif split == "val":
                    validation_guesser_accuracy.append(guesser_accuracy)
                    val_decision_loss = torch.cat(
                        [val_decision_loss, decider_loss.data / batch_size]
                    )
                    val_guesser_loss = torch.cat([val_guesser_loss, guesser_loss.data])
                    val_total_loss = torch.cat([val_total_loss, loss.data])

        if exp_config["save_models"]:
            model_file = os.path.join(
                model_dir,
                "".join(["model_ensemble_", args.bin_name, "_E_", str(epoch)]),
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "val_accuracies": val_accuracies,
                },
                model_file,
            )

        print(
            "Epoch %03d, Time taken %.3f, Total Training Loss %.4f, Total Validation Loss %.4f"
            % (
                epoch,
                time() - start,
                torch.mean(train_total_loss),
                torch.mean(val_total_loss),
            )
        )
        print(
            "Validation Loss:: Decider %.3f, Guesser %.3f"
            % (torch.mean(val_decision_loss), torch.mean(val_guesser_loss))
        )
        print("Training Accuracy:: Guesser %.3f" % (np.mean(training_guesser_accuracy)))
        print(
            "Validation Accuracy::  Guesser %.3f"
            % (np.mean(validation_guesser_accuracy))
        )
        val_accuracies.append(np.mean(validation_guesser_accuracy))

        if args.exp_tracker is not None:
            wandb.log(
                {
                    "Guesser Training Loss": torch.mean(train_guesser_loss),
                    "Guesser Validation Loss": torch.mean(val_guesser_loss),
                    "Guesser Training Accuracy": torch.mean(
                        torch.tensor(training_guesser_accuracy)
                    ),
                    "Guesser Validation Accuracy": torch.mean(
                        torch.tensor(validation_guesser_accuracy)
                    ),
                }
            )

        if exp_config["save_models"]:
            print("Saved model to %s" % (model_file))

    best_epoch = np.argmax(val_accuracies)
    best_model_file = os.path.join(
        model_dir,
        "".join(["model_ensemble_", args.bin_name, "_E_", str(best_epoch)]),
    )
    print(f"Best model: {best_model_file}")

    if args.test_data_dir is not None:
        dataset_args["data_dir"] = args.test_data_dir
        print("Evaluating over test data using best model")
        test_model(
            model_type="blind",
            best_ckpt=best_model_file,
            dataset_args=dataset_args,
            ensemble_args=ensemble_args,
            optimizer_args=optimizer_args,
            exp_config=exp_config,
        )
