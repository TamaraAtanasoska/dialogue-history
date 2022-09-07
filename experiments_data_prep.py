import gzip
import json
import random
from typing import Optional


def reverse_dialogues_in_json(folder: str, which_set: str, out_dir: str,
                              additional_address: Optional[str] = "",
                              additional_name: Optional[str] = "", ):
    file = f'{folder}{additional_address}/guesswhat{additional_name}.{which_set}.jsonl.gz'
    out_file = f'{out_dir}{additional_address}/guesswhat{additional_name}_reversed.{which_set}.jsonl.gz'

    with gzip.open(file) as f, gzip.open(out_file, 'wb') as out:
        for line in f:
            line = line.decode("utf-8")
            game = json.loads(line.strip('\n'))
            game['qas'].reverse()
            data = (json.dumps(line))
            out.write(json.loads(data).encode())


def remove_last_turn_from_dialogues_in_json(folder: str, which_set: str, out_dir: str,
                                            additional_address: Optional[str] = "",
                                            additional_name: Optional[str] = "", ):
    file = f'{folder}{additional_address}/guesswhat{additional_name}.{which_set}.jsonl.gz'
    out_file = f'{out_dir}{additional_address}/guesswhat{additional_name}_no_last_turn.{which_set}.jsonl.gz'

    with gzip.open(file) as f, gzip.open(out_file, 'wb') as out:
        for line in f:
            line = line.decode("utf-8")
            game = json.loads(line.strip('\n'))
            game['qas'] = game['qas'][:-1]
            data = (json.dumps(line))
            out.write(json.loads(data).encode())


def shuffle_dialogues_in_json(folder: str, which_set: str, out_dir: str,
                              additional_address: Optional[str] = "",
                              additional_name: Optional[str] = "", ):
    file = f'{folder}{additional_address}/guesswhat{additional_name}.{which_set}.jsonl.gz'
    out_file = f'{out_dir}{additional_address}/guesswhat{additional_name}_shuffled.{which_set}.jsonl.gz'

    with gzip.open(file) as f, gzip.open(out_file, 'wb') as out:
        for line in f:
            line = line.decode("utf-8")
            game = json.loads(line.strip('\n'))
            random.shuffle(game['qas'])
            data = (json.dumps(line))
            out.write(json.loads(data).encode())


def remove_raw_category_in_json(folder: str, which_set: str, out_dir: str, remove_id: Optional[bool] = True,
                                additional_address: Optional[str] = "",
                                additional_name: Optional[str] = "", ):
    file = f'{folder}{additional_address}/guesswhat{additional_name}.{which_set}.jsonl.gz'
    out_file = f'{out_dir}{additional_address}/guesswhat{additional_name}_without_raw_category{"_no_category_id" if remove_id else ""}.{which_set}.jsonl.gz'

    with gzip.open(file) as f, gzip.open(out_file, 'wb') as out:
        for line in f:
            line = line.decode("utf-8")
            game = json.loads(line.strip('\n'))
            for object_id in game['objects']:
                game['objects'][object_id]['category'] = 'no_category'
                if remove_id:
                    game['objects'][object_id]['category_id'] = 1
            data = (json.dumps(line))
            out.write(json.loads(data).encode())


def main():
    data_folder = "./model-repos/guesswhat/data"
    no_cat_no_id = "/experiments/no-category-no-id"
    no_cat = "/experiments/no-category"
    for data_set in ['train', 'valid', 'test']:
        remove_raw_category_in_json(data_folder, data_set, data_folder + no_cat_no_id)
        remove_raw_category_in_json(data_folder, data_set, data_folder + no_cat, remove_id=False)

    reverse_dialogues_in_json(data_folder, "test", data_folder + "/experiments")
    remove_last_turn_from_dialogues_in_json(data_folder, "test", data_folder + "/experiments")
    shuffle_dialogues_in_json(data_folder, "test", data_folder + "/experiments")

    for name, address in [('_without_raw_category', no_cat),
                          ('_without_raw_category_no_category_id', no_cat_no_id)]:
        reverse_dialogues_in_json(data_folder, "test",  data_folder, additional_name=name, additional_address=address)
        remove_last_turn_from_dialogues_in_json(data_folder, "test",  data_folder,
                                                additional_name=name,
                                                additional_address=address)
        shuffle_dialogues_in_json(data_folder, "test",  data_folder, additional_name=name, additional_address=address)


if __name__ == '__main__':
    main()
