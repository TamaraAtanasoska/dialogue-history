import gzip
import json


def reverse_dialogues_in_json(folder: str, which_set: str, out_dir: str):
    file = '{}/guesswhat.{}.jsonl.gz'.format(folder, which_set)
    out_file = '{}/guesswhat_reversed.{}.jsonl.gz'.format(out_dir, which_set)

    with gzip.open(file) as f, gzip.open(out_file, 'wb') as out:
        for line in f:
            line = line.decode("utf-8")
            game = json.loads(line.strip('\n'))
            game['qas'].reverse()
            out.write(str(json.dumps(line)).encode())
            out.write(b'\n')


def remove_last_turn_from_dialogues_in_json(folder: str, which_set: str, out_dir: str):
    file = '{}/guesswhat.{}.jsonl.gz'.format(folder, which_set)
    out_file = '{}/guesswhat_no_last_turn.{}.jsonl.gz'.format(out_dir, which_set)

    with gzip.open(file) as f, gzip.open(out_file, 'wb') as out:
        for line in f:
            line = line.decode("utf-8")
            game = json.loads(line.strip('\n'))
            game['qas'] = game['qas'][:-1]
            out.write(str(json.dumps(line)).encode())
            out.write(b'\n')


def main():
    # reverse_dialogues_in_json("data", "train", "data/experiments")
    for data_set in ['train', 'valid', 'test']:
        reverse_dialogues_in_json("data", data_set, "data/experiments")
        remove_last_turn_from_dialogues_in_json("data", data_set, "data/experiments")


if __name__ == '__main__':
    main()


