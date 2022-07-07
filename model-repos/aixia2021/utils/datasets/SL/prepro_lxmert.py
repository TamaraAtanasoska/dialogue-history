import argparse
import gzip
import json
import os

from nltk.tokenize import TweetTokenizer

from utils.image_utils import get_spatial_feat


def create_data_file(data_dir, data_file, data_args, vocab_file_name, split='train'):
    '''Creates the training/test/val data given dataset file in *.jsonl.gz format.

    Parameters
    ----------
    data_dir : str
        Directory to read the data and dump the training data created
    data_file : str
        Name of the *.jsonl.gz data file
    data_args : dict
        'successful_only' : bool. Checks what type of games to be included.
        'max_no_objects' : int. Number required for padding of objects in target list for Guesser.
        'max_q_length' : int. Max number of words that QGen can use to ask next question
        'max_src_length' : int. Max number of words that can be present in the dialogue history
        'max_no_qs' : int. Max number of questions that a gamme can have to be included in the data
        'data_paths' : str?. Added by ravi for different file name than default. More details to be added by ravi.
    vocab_file_name : str
        vocabulary file name. This file should have 'word2i' and 'i2word'
    split : str
        Split of the data file
    '''
    path = os.path.join(data_dir, data_file)
    successful_only = data_args['successful_only']

    tmp_key = split + '_process_file'

    if tmp_key in data_args['data_paths']:
        data_file_name = data_args['data_paths'][tmp_key]
    else:
        if successful_only:
            data_file_name = 'n2n_' + split + '_successful_data_lxmert.json'
        else:
            data_file_name = 'n2n_' + split + '_all_data_lxmert.json'

    print('Creating New ' + data_file_name + ' File.')

    category_pad_token = 0  # TODO Add this to config.json
    decidermask_pad_token = -1  # TODO Add this to config.json
    max_no_objects = data_args['max_no_objects']
    max_q_length = data_args['max_q_length']
    max_src_length = data_args['max_src_length']
    max_no_qs = data_args['max_no_qs']
    no_spatial_feat = 8  # TODO Add this to config.json

    tknzr = TweetTokenizer(preserve_case=False)
    n2n_data = dict()
    _id = 0

    # load or create new vocab
    with open(os.path.join(data_dir, vocab_file_name), 'r') as file:
        vocab = json.load(file)
        word2i = vocab['word2i']
        i2word = vocab['i2word']

    ans2tok = {'Yes': word2i['<yes>'],
               'No': word2i['<no>'],
               'N/A': word2i['<n/a>']}

    start = '<start>'
    # stop = '<stop>'

    with gzip.open(path) as file:

        for json_game in file:
            game = json.loads(json_game.decode('utf-8'))

            if successful_only:
                if not game['status'] == 'success':
                    continue

            if len(game['qas']) > max_no_qs:
                continue

            objects = list()
            object_ids = list()  # These are added for crop features
            spatials = list()
            bboxes = list()
            target = int()
            target_cat = int()
            for i, o in enumerate(game['objects']):
                object = game['objects'][o]
                objects.append(object['category_id'])
                object_ids.append(object['object_id'])
                spatials.append(get_spatial_feat(bbox=object['bbox'], im_width=game['picture']['width'],
                                                 im_height=game['picture']['height']))

                if object['object_id'] == game['object_id']:
                    target = i
                    target_cat = object['category_id']
                    bboxes.append(object['bbox'])
            # pad objects, spatials and bboxes
            objects.extend([category_pad_token] * (max_no_objects - len(objects)))
            object_ids.extend([0] * (max_no_objects - len(object_ids)))
            spatials.extend([[0] * no_spatial_feat] * (max_no_objects - len(spatials)))

            # dialogue history and target question
            src = list()
            src_lengths = list()

            src_raw = ''

            for i, qa in enumerate(game['qas']):
                if i != 0:
                    # remove padding from previous target and current source
                    src_unpad = src[:src.index(word2i['<padding>'])] if word2i['<padding>'] in src else src
                    target_q_unpad = target_q[:target_q.index(word2i['<padding>'])] if word2i[
                                                                                           '<padding>'] in target_q else target_q
                    src = src_unpad + target_q_unpad + [ans2tok[answer]]
                    src_raw = src_raw + target_q_raw + ' ' + answer + ' '
                else:
                    src = [word2i[start]]
                src_lengths.append(len(src))

                q_tokens = tknzr.tokenize(qa['q'])
                answer = qa['a']
                target_q_raw = qa['q']

                target_q = [word2i[w] if w in word2i else word2i['<unk>'] for w in q_tokens]
                src_q = [word2i[start]] + [word2i[w] if w in word2i else word2i['<unk>'] for w in q_tokens]

                # All decider targets here are 0
                n2n_data[_id] = dict()
                n2n_data[_id]['tgt_len'] = min(len(target_q), max_q_length)
                n2n_data[_id]['history_len'] = min(len(src), max_src_length)
                history_q_lens = src_lengths[:]  # Deep copy
                len_hql = len(history_q_lens)
                history_q_lens.extend([0] * ((max_no_qs + 1) - len(
                    history_q_lens)))  # +1 because of start token is consiidered as first question
                n2n_data[_id]['history_q_lens'] = history_q_lens
                n2n_data[_id]['len_hql'] = len_hql
                target_q.extend([word2i['<padding>']] * (max_q_length - len(target_q)))
                src_q.extend([word2i['<padding>']] * (max_q_length - len(src_q)))
                src.extend([word2i['<padding>']] * (max_src_length - len(src)))
                n2n_data[_id]['history'] = src[:max_src_length]
                n2n_data[_id]['history_raw'] = src_raw.strip()
                n2n_data[_id]['src_q'] = src_q[:max_q_length]
                n2n_data[_id]['target_q'] = target_q[:max_q_length]
                n2n_data[_id]['decider_tgt'] = 0
                decider_mask = [n2n_data[_id]['decider_tgt']] * len_hql
                decider_mask.extend([decidermask_pad_token] * ((max_no_qs + 1) - len_hql))
                n2n_data[_id]['decider_mask'] = decider_mask
                n2n_data[_id]['objects'] = objects
                n2n_data[_id]['object_ids'] = object_ids
                n2n_data[_id]['spatials'] = spatials
                n2n_data[_id]['target_obj'] = target
                n2n_data[_id]['target_cat'] = target_cat
                n2n_data[_id][
                    'bboxes'] = bboxes  # Change in v2 only target bbox is included as everything is not required
                n2n_data[_id]['game_id'] = str(game['dialogue_id'])
                n2n_data[_id]['image_file'] = game['picture']['file_name']
                n2n_data[_id]['image_url'] = game['picture']['coco_url']
                _id += 1

            src_unpad = src[:src.index(word2i['<padding>'])] if word2i['<padding>'] in src else src
            target_q_unpad = target_q[:target_q.index(word2i['<padding>'])] if word2i[
                                                                                   '<padding>'] in target_q else target_q
            src = src_unpad + target_q_unpad + [ans2tok[answer]]
            src_q = [0]
            target_q = [0]
            src_lengths.append(len(src))
            src_raw = src_raw + target_q_raw + ' ' + answer + ' '

            # Decider target 1
            n2n_data[_id] = dict()
            n2n_data[_id]['tgt_len'] = min(len(target_q), max_q_length)
            n2n_data[_id]['history_len'] = min(len(src), max_src_length)
            history_q_lens = src_lengths[:]  # Deep copy
            len_hql = len(history_q_lens)
            history_q_lens.extend([0] * ((max_no_qs + 1) - len(
                history_q_lens)))  # +1 because of start token is consiidered as first question
            n2n_data[_id]['history_q_lens'] = history_q_lens
            n2n_data[_id]['len_hql'] = len_hql
            target_q.extend([word2i['<padding>']] * (max_q_length - len(target_q)))
            src_q.extend([word2i['<padding>']] * (max_q_length - len(src_q)))
            src.extend([word2i['<padding>']] * (max_src_length - len(src)))
            n2n_data[_id]['history'] = src[:max_src_length]
            n2n_data[_id]['history_raw'] = src_raw.strip()
            n2n_data[_id]['src_q'] = src_q[:max_q_length]
            n2n_data[_id]['target_q'] = target_q[:max_q_length]
            n2n_data[_id]['decider_tgt'] = 1
            decider_mask = [0] * (len_hql - 1) + [
                n2n_data[_id]['decider_tgt']] * 1  # Because only the last target is guess
            decider_mask.extend([decidermask_pad_token] * ((max_no_qs + 1) - len_hql))
            n2n_data[_id]['decider_mask'] = decider_mask
            n2n_data[_id]['objects'] = objects
            n2n_data[_id]['object_ids'] = object_ids
            n2n_data[_id]['spatials'] = spatials
            n2n_data[_id]['target_obj'] = target
            n2n_data[_id]['target_cat'] = target_cat
            n2n_data[_id]['bboxes'] = bboxes  # Change in v2 only target bbox is included as everything is not required
            n2n_data[_id]['game_id'] = str(game['dialogue_id'])
            n2n_data[_id]['image_file'] = game['picture']['file_name']
            n2n_data[_id]['image_url'] = game['picture']['coco_url']
            _id += 1

    n2n_data_path = os.path.join(data_dir, data_file_name)
    with open(n2n_data_path, 'w') as f:
        json.dump(n2n_data, f)

    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="data/test/experiments/reverse_history",
                        help='Target Data Directory to store the vocab file')
    parser.add_argument("-data_file", type=str, default="guesswhat.test.jsonl.gz", help='Guesswhat train data file')
    parser.add_argument("-vocab_file", type=str, default='vocab.json', help='Vocabulary file')
    parser.add_argument("-split", type=str, default='test', help='Split name')

    args = parser.parse_args()
    split = args.split
    data_dir = args.data_dir
    data_file = args.data_file
    vocab_file = args.vocab_file

    data_args = {
        'max_src_length': 200,
        'max_q_length': 30,
        'max_no_objects': 20,
        'max_no_qs': 10,
        'successful_only': True,
        'data_paths': ''
    }

    create_data_file(data_dir=data_dir, data_file=data_file, data_args=data_args, vocab_file_name=vocab_file,
                     split=split)
