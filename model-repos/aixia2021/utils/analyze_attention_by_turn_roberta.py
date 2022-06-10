import argparse
import csv
import string
from collections import defaultdict, Counter

import nltk
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaTokenizer

from models.BERTEnsemble import BERTEnsemble
from train.SL.parser import preprocess_config
from utils.datasets.SL.N2NBERTDataset import N2NBERTDataset
from utils.model_loading import load_model

# TODO Make this capitalised everywhere to inform it is a global variable
use_cuda = torch.cuda.is_available()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="data", help="Data Directory")
    parser.add_argument("-config", type=str, default="config/SL/config_bert.json")
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

    model = BERTEnsemble(**ensemble_args)
    print("Loading model: {}".format(args.load_bin_path))
    model = load_model(model, args.load_bin_path, use_dataparallel=use_cuda)
    model.eval()

    if use_cuda:
        model.cuda()
        # model = DataParallel(model)
    print(model)

    dataset_test = N2NBERTDataset(split="test", **dataset_args, num_turns=5, complete_only=True)

    dataloader = DataLoader(
        dataset=dataset_test,
        batch_size=64,
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

    tokenizer = RobertaTokenizer.from_pretrained(
        "roberta-base"
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

            guesser_out = model(
                src_q=sample["src_q"],
                tgt_len=sample["tgt_len"],
                spatials=sample["spatials"],
                objects=sample["objects"],
                mask_select=1,
                target_cat=sample["target_cat"],
                history_raw=sample["history_raw"]
            )

            strange_character = tokenizer.tokenize(" hello")[0][0]

            for layer in range(12):
                for head in range(12):
                    # has shape (batch_size, num_heads, 200, 200)
                    lang2lang_attention_probs = model.module.encoder_attentions[layer].detach().cpu().numpy()

                    # has shape (batch_size, 200, 200)
                    head_lang2lang_attention_probs = lang2lang_attention_probs[:, head, :, :]

                    for datapoint in range(len(sample["game_id"])):
                        tokenized_turn = tokenizer.tokenize(sample["history_raw"][datapoint].lower())
                        tokenized_turn = ["<s>"] + tokenized_turn + ["</s>"]
                        tokenized_turn = [token[1:] if token[0] == strange_character else token for token in tokenized_turn]

                        tokens_turns = []
                        current_turn = 0
                        already_inserted_answer = False
                        for token_index, token in enumerate(tokenized_turn):
                            token = token.lower()
                            if token == "<s>":
                                tokens_turns.append(current_turn)
                                current_turn += 1
                            elif token == "</s>":
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

        with open("turn_probs_roberta_scratch_max_reversed.csv", mode="w") as out_file:
            writer = csv.writer(out_file)
            writer.writerow(["Layer", "Turn", "Attention"])
            for layer in turn_probs:
                turn_probs_per_layer = defaultdict(list)
                for head in turn_probs[layer]:
                    for turn in range(7):
                        turn_probs_per_layer[turn].append(np.mean(turn_probs[layer][head][turn]))
                for turn in range(7):
                    writer.writerow([layer, turn, np.mean(turn_probs_per_layer[turn])])

        print("The end :-(")
