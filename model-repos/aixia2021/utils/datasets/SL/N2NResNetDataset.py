import json
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.create_subset import create_subset
from utils.datasets.SL.prepro import create_data_file


class N2NResNetDataset(Dataset):
    def __init__(self, split='train', **kwargs):
        # data_dir, data_file, data_args, vocab_file_name,
        self.data_args = kwargs

        tmp_key = split + '_process_file'

        self.img_dir = os.path.join(self.data_args['data_paths']['image_path'], split)

        if tmp_key in self.data_args['data_paths']:
            data_file_name = self.data_args['data_paths'][tmp_key]
        else:
            if self.data_args['successful_only']:
                data_file_name = 'n2n_' + split + '_successful_data.json'
            else:
                data_file_name = 'n2n_' + split + '_all_data.json'

        if self.data_args['new_data'] or not os.path.isfile(os.path.join(self.data_args['data_dir'], data_file_name)):
            create_data_file(
                data_dir=self.data_args['data_dir'],
                data_file=self.data_args['data_paths'][split],
                data_args=self.data_args,
                vocab_file_name=self.data_args['data_paths']['vocab_file'],
                split=split
            )

        if self.data_args['my_cpu']:
            if not os.path.isfile(os.path.join(self.data_args['data_dir'], 'subset_' + data_file_name)):
                create_subset(data_dir=self.data_args['data_dir'], dataset_file_name=data_file_name, split=split)

        if self.data_args['my_cpu']:
            with open(os.path.join(self.data_args['data_dir'], 'subset_' + data_file_name), 'r') as f:
                self.n2n_data = json.load(f)
        else:
            with open(os.path.join(self.data_args['data_dir'], data_file_name), 'r') as f:
                self.n2n_data = json.load(f)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

    def __len__(self):
        return len(self.n2n_data)

    def __getitem__(self, idx):
        if not type(idx) == str:
            idx = str(idx)

        tmp_img_path = os.path.join(self.img_dir, self.n2n_data[idx]['image_file'])
        if os.path.isfile(tmp_img_path):
            img_path = tmp_img_path
        else:
            # Taking care if image is stored as in MS-COCO directory structure
            tmp_img_path = os.path.join(self.data_args['data_paths']['image_path'], 'train2014',
                                        self.n2n_data[idx]['image_file'])
            if os.path.isfile(tmp_img_path):
                img_path = tmp_img_path
            else:
                tmp_img_path = os.path.join(self.data_args['data_paths']['image_path'], 'val2014',
                                            self.n2n_data[idx]['image_file'])
                if os.path.isfile(tmp_img_path):
                    img_path = tmp_img_path
                else:
                    print('Something wrong with image path')

        ImgTensor = self.transform(Image.open(img_path).convert('RGB'))
        _data = dict()
        _data['history'] = np.asarray(self.n2n_data[idx]['history'])
        _data['history_len'] = self.n2n_data[idx]['history_len']
        _data['src_q'] = np.asarray(self.n2n_data[idx]['src_q'])
        _data['target_q'] = np.asarray(self.n2n_data[idx]['target_q'])
        _data['tgt_len'] = self.n2n_data[idx]['tgt_len']
        _data['decider_tgt'] = int(self.n2n_data[idx]['decider_tgt'])
        _data['objects'] = np.asarray(self.n2n_data[idx]['objects'])
        _data['objects_mask'] = np.asarray(
            1 - np.equal(self.n2n_data[idx]['objects'], np.zeros(len(self.n2n_data[idx]['objects']))))
        _data['spatials'] = np.asarray(self.n2n_data[idx]['spatials'])
        _data['target_obj'] = self.n2n_data[idx]['target_obj']
        _data['target_cat'] = self.n2n_data[idx]['target_cat']
        _data['image'] = ImgTensor
        _data['game_id'] = self.n2n_data[idx]['game_id']
        _data['history_q_lens'] = np.asarray(self.n2n_data[idx]['history_q_lens'])
        _data['decider_mask'] = np.asarray(self.n2n_data[idx]['decider_mask'])
        _data['len_hql'] = self.n2n_data[idx]['len_hql']
        _data['bboxes'] = np.asarray(self.n2n_data[idx]['bboxes'])
        _data['image_url'] = self.n2n_data[idx]['image_url']

        return _data
