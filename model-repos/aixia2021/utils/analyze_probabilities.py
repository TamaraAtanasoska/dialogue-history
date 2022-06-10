import collections
import json

import numpy as np
from scipy.stats import entropy

if __name__ == "__main__":
    with open("../roberta_save_guess_probs.json") as in_file:
        save_guess_probs = json.load(in_file)

    save_target_prob = collections.defaultdict(list)
    save_target_dist = collections.defaultdict(list)
    for k, v in save_guess_probs.items():
        if len(v) <= 2:
        # if len(v) <= 3:
        # if len(v) <= 2:
            continue
        target_prob = 0
        prob_dist=[]
        for idx in range(1, len(v)):
            # if v[str(idx)]["obj_mask"] != 5:
            #     continue
        # for idx in range(1, len(v)-1):
        # for idx in range(len(v)-2, len(v)):
            if idx == 1:
            # if idx == len(v)-2:
                idx = str(idx)
                target_prob = v[idx]['probs'][v[idx]['target']]
                prob_dist = v[idx]['probs'][:v[idx]['obj_mask']]
            else:
                idx = str(idx)
                new_target_prob = v[idx]['probs'][v[idx]['target']]
                new_prob_dist = v[idx]['probs'][:v[idx]['obj_mask']]
                save_target_prob[v[idx]['ans']].append(new_target_prob-target_prob)
                save_target_dist[v[idx]['ans']].append(entropy(prob_dist, new_prob_dist))
                target_prob = new_target_prob
                prob_dist = new_prob_dist

    print("GDSE-SL")
    for k, v in save_target_prob.items():
        print("Change in probability assigned to the target object after '{}' answer:\t\t{}".format(k, round(np.mean(v), 3)))
    print()
    for k, v in save_target_dist.items():
        print("Change in overall probability distribution (KL divergence) after '{}' answer:\t{}".format(k, round(np.mean(v), 3)))
