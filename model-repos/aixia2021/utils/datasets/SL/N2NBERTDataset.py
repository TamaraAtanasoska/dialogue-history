import json
import os
import random
from copy import deepcopy

import h5py
import numpy as np
from torch.utils.data import Dataset


# from utils.datasets.SL.prepro_lxmert import create_data_file


class N2NBERTDataset(Dataset):
    def __init__(self, split='train', split_turns=False, add_sep=False, create_subset=None, with_objects_feat=False, num_turns=None, game_ids=None, complete_only=False, **kwargs):
        self.data_args = kwargs
        self.with_objects_feat = with_objects_feat

        if with_objects_feat:
            objects_feat_file = os.path.join(self.data_args['data_dir'], self.data_args['data_paths']['ResNet']['objects_features'] )
            objects_feat_mapping_file = os.path.join(self.data_args['data_dir'], self.data_args['data_paths']['ResNet']['objects_features_index'] )
            self.objects_vf = h5py.File(objects_feat_file, 'r')['objects_features']

            with open(objects_feat_mapping_file, 'r') as file_v:
                self.objects_feat_mapping = json.load(file_v)

        tmp_key = split + "_process_file"

        if tmp_key in self.data_args['data_paths']:
            data_file_name = self.data_args['data_paths'][tmp_key]
        else:
            if self.data_args['successful_only']:
                data_file_name = 'n2n_'+split+'_successful_data_lxmert.json'
            else:
                data_file_name = 'n2n_'+split+'_all_data_lxmert.json'

        if self.data_args['new_data'] or not os.path.isfile(os.path.join(self.data_args['data_dir'], data_file_name)):
            create_data_file(
                data_dir=self.data_args['data_dir'],
                data_file=self.data_args['data_paths'][split],
                data_args=self.data_args,
                vocab_file_name=self.data_args['data_paths']['vocab_file'],
                split=split
            )

        if self.data_args['my_cpu']:
            if not os.path.isfile(os.path.join(self.data_args['data_dir'], 'subset_'+split+'_lxmert.json')):
                create_subset(data_dir=self.data_args['data_dir'], dataset_file_name=data_file_name, split=split)

        if self.data_args['my_cpu']:
            with open(os.path.join(self.data_args['data_dir'], 'subset_'+split+'_lxmert.json'), 'r') as f:
                self.n2n_data = json.load(f)
        else:
            with open(os.path.join(self.data_args['data_dir'], data_file_name), 'r') as f:
                self.n2n_data = json.load(f)

        if self.data_args['breaking']:
            n2n_data_filtered = {}
            _id = 0
            for example_id, example in self.n2n_data.items():
                if example["image_file"].split(".")[0] in self.data_args["imgid2fasterRCNNfeatures"]:
                    n2n_data_filtered[str(_id)] = example
                    _id += 1
            self.n2n_data = n2n_data_filtered

        if create_subset is not None:
            print("Taking only dialogs belonging to a sampled subset of IDs...")
            subset_filename = "data/subset_{}.txt".format(create_subset)
            if not os.path.exists(subset_filename):
                print("Writing file {}".format(subset_filename))
                all_game_ids = {v["game_id"] for _, v in self.n2n_data.items()}
                k = int(len(all_game_ids) * create_subset)
                sampled_game_ids = set(random.sample(all_game_ids, k))
                with open(subset_filename, mode="w") as out_file:
                    for id in sampled_game_ids:
                        out_file.write(id + "\n")
            else:
                print("Reading file {}".format(subset_filename))
                with open(subset_filename) as in_file:
                    sampled_game_ids = set([x.strip() for x in in_file.readlines()])

            print("Filtering...")
            filtered_n2n_data = {}
            _id = 0
            for k, v in self.n2n_data.items():
                if v["game_id"] in sampled_game_ids:
                    filtered_n2n_data[str(_id)] = v
                    _id += 1
            self.n2n_data = filtered_n2n_data

        if game_ids is not None:
            game_ids = [str(x) for x in game_ids]
            print("Taking only dialogs belonging to a given subset of IDs...")
            filtered_n2n_data = {}
            _id = 0
            for k, v in self.n2n_data.items():
                if v["game_id"] in game_ids:
                    filtered_n2n_data[str(_id)] = v
                    _id += 1
            self.n2n_data = filtered_n2n_data

        if num_turns is not None:
            print("Taking only dialogs having {} turns...".format(num_turns))
            filtered_n2n_data = {}
            _id = 0
            for k, v in self.n2n_data.items():
                if sum([1 for x in v["history_q_lens"] if x != 0]) == num_turns + 1:
                    filtered_n2n_data[str(_id)] = v
                    _id += 1
            self.n2n_data = filtered_n2n_data

        if complete_only or split_turns:
            print("Taking only complete dialogs...")
            filtered_n2n_data = {}
            _id = 0
            for k, v in self.n2n_data.items():
                if v["decider_tgt"] == 1:
                    filtered_n2n_data[str(_id)] = v
                    _id += 1
            self.n2n_data = filtered_n2n_data

        if add_sep:
            print("Adding SEP token to dialogs...")
            filtered_n2n_data = {}
            _id = 0
            for k, v in self.n2n_data.items():
                h = v["history_raw"]
                turn = 0
                new_h = ""
                history_turns = []
                new_turn = False
                tokens = h.replace("?", " ?").split()
                for token_index, token in enumerate(tokens):
                    new_h += token + " "
                    if new_turn:
                        history_turns.append(new_h.strip() + " [SEP]")
                        new_h = ""
                        new_turn = False
                    if token == "?" and tokens[token_index + 1].lower() in ["yes", "no", "n/a"]:
                        new_turn = True
                        turn += 1

                new_history_raw = " ".join(history_turns)[:-6]

                v["history_raw"] = new_history_raw
                filtered_n2n_data[str(_id)] = v
                _id += 1

            self.n2n_data = filtered_n2n_data

        if split_turns:
            print("Splitting dialogs by turn...")
            filtered_n2n_data = {}
            _id = 0
            for k, v in self.n2n_data.items():
                h = v["history_raw"]
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

                for turn in history_turns:
                    new_dialog = deepcopy(v)
                    new_dialog["history_raw"] = turn
                    filtered_n2n_data[str(_id)] = new_dialog
                    _id += 1
            self.n2n_data = filtered_n2n_data

    def __len__(self):
        return len(self.n2n_data)

    def __getitem__(self, idx):
        if not type(idx) == str:
            idx = str(idx)

        _data = dict()
        _data['history'] = np.asarray(self.n2n_data[idx]['history'])
        _data['history_len'] = self.n2n_data[idx]['history_len']
        # _data['history_q_lens'] = self.n2n_data[idx]['history_q_lens']
        _data['src_q'] = np.asarray(self.n2n_data[idx]['src_q'])
        _data['target_q'] = np.asarray(self.n2n_data[idx]['target_q'])
        _data['tgt_len'] = self.n2n_data[idx]['tgt_len']
        _data['decider_tgt'] = int(self.n2n_data[idx]['decider_tgt'])
        _data['objects'] = np.asarray(self.n2n_data[idx]['objects'])
        _data['objects_mask'] = np.asarray(1-np.equal(self.n2n_data[idx]['objects'], np.zeros(len(self.n2n_data[idx]['objects']))))
        _data['spatials'] = np.asarray(self.n2n_data[idx]['spatials'])
        _data['target_obj'] = self.n2n_data[idx]['target_obj']
        _data['target_cat'] = self.n2n_data[idx]['target_cat']
        _data['game_id'] = self.n2n_data[idx]['game_id']
        _data['bboxes'] = np.asarray(self.n2n_data[idx]['bboxes'])
        _data['image_url'] = self.n2n_data[idx]['image_url']
        _data['image_file'] = self.n2n_data[idx]['image_file']
        _data['history_raw'] = self.n2n_data[idx]['history_raw']

        # Load object features
        if self.with_objects_feat:
            objects_feat_id = self.objects_feat_mapping[self.n2n_data[idx]['game_id']]
            objects_feat = self.objects_vf[objects_feat_id]
            _data["objects_feat"] = objects_feat

        return _data
