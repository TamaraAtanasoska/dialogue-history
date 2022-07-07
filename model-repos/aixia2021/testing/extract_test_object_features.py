import argparse
import gzip
import json
import os

import h5py
import numpy as np
from PIL import Image
from torchvision import transforms

from models.CNN import ResNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-image_dir", type=str, default="/home/users/bverma/project/sync/al/data/images")
    parser.add_argument("-test_set", type=str, default="data/test/experiments/reverse_history/guesswhat.test.jsonl.gz")
    parser.add_argument("-objects_features_index_path", type=str,
                        default="data/test/experiments/reverse_history/objects_features_index_example.json")
    parser.add_argument("-objects_features_path", type=str,
                        default="data/test/experiments/reverse_history/objects_features_example.h5")
    args = parser.parse_args()

    games = []
    print("Loading file: {}".format(args.test_set))
    with gzip.open(args.test_set) as file:
        for json_game in file:
            games.append(json.loads(json_game.decode("utf-8")))

    model = ResNet()
    model.eval()
    model.cuda()

    avg_img_features = np.zeros((len(games), 20, 2048))
    game_id2pos = {}

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )

    for game_index, game in enumerate(games):
        print("\rProcessing image [{}/{}]".format(game_index, len(games)), end="")

        image = Image.open(os.path.join(args.image_dir, game["picture"]["file_name"])).convert("RGB")
        game_id2pos[str(game["dialogue_id"])] = game_index

        for object_index, object_id in enumerate(game["objects"]):
            object = game["objects"][object_id]
            bbox = object["bbox"]
            cropped_image = image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
            cropped_image_tensor = transform(cropped_image)
            cropped_image_tensor = cropped_image_tensor.view(1, 3, 224, 224)
            conv_features, feat = model(cropped_image_tensor.cuda())
            avg_img_features[game_index][object_index] = feat.cpu().data.numpy()

    print("Saving file: {}".format(args.objects_features_path))
    objects_features_h5 = h5py.File(args.objects_features_path, "w")
    objects_features_h5.create_dataset(name="objects_features", dtype="float32", data=avg_img_features)
    objects_features_h5.close()

    print("Saving file: {}".format(args.objects_features_index_path))
    with open(args.objects_features_index_path, mode="w") as out_file:
        json.dump(game_id2pos, out_file)
