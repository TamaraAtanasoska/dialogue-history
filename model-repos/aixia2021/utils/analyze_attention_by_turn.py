import argparse
import csv
import json
import string
from collections import defaultdict, Counter

import nltk
import numpy as np
import sharearray
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lxmert.src.lxrt.tokenization import BertTokenizer
from models.LXMERTEnsembleGuesserOnly import LXMERTEnsembleGuesserOnly
from train.SL.parser import preprocess_config
from utils.datasets.SL.N2NLXMERTDataset import N2NLXMERTDataset
from utils.model_loading import load_model

# TODO Make this capitalised everywhere to inform it is a global variable
use_cuda = torch.cuda.is_available()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="data", help="Data Directory")
    parser.add_argument("-config", type=str, default="config/SL/config_bert_scratch.json")
    parser.add_argument("-my_cpu", action="store_true")
    parser.add_argument("-breaking", action="store_true")
    parser.add_argument("-exp_name", type=str)
    parser.add_argument("-bin_name", type=str)
    parser.add_argument("-num_regions", type=int)
    parser.add_argument("-load_bin_path", type=str)
    parser.add_argument("--preloaded", type=bool, default=False)
    args = parser.parse_args()
    print(args.exp_name)

    ensemble_args, dataset_args, optimizer_args, exp_config = preprocess_config(args)

    float_tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    torch.manual_seed(exp_config["seed"])
    if use_cuda:
        torch.cuda.manual_seed_all(exp_config["seed"])

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

    model = LXMERTEnsembleGuesserOnly(**ensemble_args)
    print("Loading model: {}".format(args.load_bin_path))
    model = load_model(model, args.load_bin_path, use_dataparallel=use_cuda)
    model.eval()

    if use_cuda:
        model.cuda()
        # model = DataParallel(model)
    print(model)

    dataset_test = N2NLXMERTDataset(split="test", **dataset_args, num_turns=5, complete_only=True, imgid2fasterRCNNfeatures=imgid2fasterRCNNfeatures)

    dataloader = DataLoader(
        dataset=dataset_test,
        batch_size=8,
        shuffle=False,
        drop_last=False,
        pin_memory=use_cuda,
        num_workers=0
    )

    softmax = nn.Softmax(dim=-1)

    word2category = {}
    with open("data/word_annotation") as in_file:
        reader = csv.reader(in_file, delimiter="\t")
        for row in reader:
            word = row[0].strip().lower()
            category = row[1].strip().lower()
            word2category[word] = category

    stopwords = set(nltk.corpus.stopwords.words("english"))
    stopwords.update(list(string.punctuation))
    stopwords.update(["yes", "no", "n", "a"])
    stopwords.remove("it")

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased",
        do_lower_case=True
    )

    # header = ["<CLS>", "<SEP>", "it", "OTHER"] + list(set((word2category.values())))
    # header2pos = {category: pos for pos, category in enumerate(header)}
    # category_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    turn_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    with torch.no_grad():
        for i_batch, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
            # if i_batch > 50:
            #     print("Breaking after processing 60 batch")
            #     break

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
                    if token == "?" and tokens[token_index + 1].lower() in ["yes", "no", "n/a"]:
                        new_turn = True
                        turn += 1
                new_history_raw.append(history_turns)

            new_history_raw = [" ".join(list(reversed(x))) for x in new_history_raw]

            decider_out, guesser_out = model(
                src_q=sample["src_q"],
                tgt_len=sample["tgt_len"],
                visual_features=sample["image"],
                spatials=sample["spatials"],
                objects=sample["objects"],
                mask_select=1,
                target_cat=sample["target_cat"],
                history_raw=sample["history_raw"],
                fasterrcnn_features=sample["FasterRCNN"]["features"],
                fasterrcnn_boxes=sample["FasterRCNN"]["boxes"]
            )

            for layer in range(5):
                for head in range(12):
                    # has shape (batch_size, num_heads, 200, 200)
                    lang2lang_attention_probs = model.module.lxrt_encoder.model.bert.encoder.x_layers[layer].lang_self_att.self.attention_probs.detach().cpu().numpy()

                    # has shape (batch_size, 200, 200)
                    head_lang2lang_attention_probs = lang2lang_attention_probs[:, head, :, :]

                    for datapoint in range(len(sample["game_id"])):
                        tokenized_turn = tokenizer.tokenize(sample["history_raw"][datapoint])
                        tokenized_turn = ["<CLS>"] + tokenized_turn + ["<SEP>"]

                        tokens_turns = []
                        current_turn = 0
                        already_inserted_answer = False
                        for token_index, token in enumerate(tokenized_turn):
                            token = token.lower()
                            if token == "<cls>":
                                tokens_turns.append(current_turn)
                                current_turn += 1
                            elif token == "<sep>":
                                tokens_turns.append(current_turn)
                            else:
                                if token == "?" and tokenized_turn[token_index+1] in ["yes", "no", "n"]:
                                    tokens_turns.append(current_turn)
                                    tokens_turns.append(current_turn)
                                    if tokenized_turn[token_index+1] == "n":
                                        tokens_turns.append(current_turn)
                                        tokens_turns.append(current_turn)
                                    current_turn += 1
                                    already_inserted_answer = True
                                elif not (token in ["yes", "no"] and tokenized_turn[token_index-1] == "?") and not (token == "n" and tokenized_turn[token_index-1] == "?"):
                                    tokens_turns.append(current_turn)

                        tokens_turns_counts = Counter(tokens_turns)

                        single_turn_probs = defaultdict(float)
                        for token_index, token in enumerate(tokenized_turn):
                            single_turn_probs[tokens_turns[token_index]] += head_lang2lang_attention_probs[datapoint][0][token_index] / tokens_turns_counts[tokens_turns[token_index]]

                        for turn, att in single_turn_probs.items():
                            turn_probs[layer][head][turn].append(att)

                        a = 0

        # with open("turn_probs_guesser_only.csv", mode="w") as out_file:
        #     writer = csv.writer(out_file)
        #     writer.writerow(["Layer", "Head", "Turn", "Attention"])
        #     for layer in turn_probs:
        #         for head in turn_probs[layer]:
        #             for turn in range(7):
        #                 writer.writerow([layer, head, turn, np.mean(turn_probs[layer][head][turn])])

        with open("turn_probs_scratch_max.csv", mode="w") as out_file:
            writer = csv.writer(out_file)
            writer.writerow(["Layer", "Turn", "Attention"])
            for layer in turn_probs:
                turn_probs_per_layer = defaultdict(list)
                for head in turn_probs[layer]:
                    for turn in range(7):
                        turn_probs_per_layer[turn].append(np.mean(turn_probs[layer][head][turn]))
                for turn in range(7):
                    writer.writerow([layer, turn, np.mean(turn_probs_per_layer[turn])])

        print("The end :-)")
