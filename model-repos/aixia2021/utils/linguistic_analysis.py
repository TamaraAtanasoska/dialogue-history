import json
from collections import Counter
from statistics import mean

import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.util import ngrams
from scipy.stats import entropy


def get_entropy_ctr(ctr):

    values = list(ctr.values())
    sum_values = float(sum(values))
    probs = [x/sum_values for x in values]
    return entropy(probs)


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def select_content_words(tokens, tag_list):
    tokens = nltk.pos_tag(tokens)
    content_tokens = []
    for t in tokens:
        if t[1] in tag_list:
            content_tokens.append(t[0])
    return content_tokens


if __name__ == '__main__':
    # gdse_file = './logs/GamePlay/model_ensemble_lxmert2020_02_16_17_50/val_E_{}_GPinference_model_ensemble_lxmert_2020_02_16_17_50.json'
    gdse_file = './logs/GamePlay/model_ensemble_lxmert2020_03_14_11_01/val_E_{}_GPinference_model_ensemble_lxmert_2020_03_14_11_01.json'
    # mixed
    # gdse_file = './logs/GamePlay/xxx2020_02_09_09_34/val_E_{}_GPinference_xxx_2020_02_09_09_34.json'

    # with open('/mnt/povobackup/clic/alberto.testoni/dialogue/VisDial-GDSE/total_questions_SUBSET_NEW.txt') as ff:
    #     train_q_newline = ff.readlines()

    print(gdse_file)

    with open('train_questions_full.txt') as ff:
        train_q_newline = ff.readlines()

    # with open('train_questions_half.txt') as ff:
    #     train_q_newline = ff.readlines()

    len_vocab = 4900

    content_words = ['NN', 'NNS', 'JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RB', 'RBR', 'RBS']

    # with open('/mnt/povobackup/clic/alberto.testoni/dialogue/VisDial-GDSE/q_per_image_id.json', 'r') as ff:
    #     q_per_image_id = json.load(ff)
    #
    # token_image_id = dict()
    # for el in q_per_image_id:
    #     token_image_id[el] = set()
    #     for round in range(10):
    #         for t in nltk.word_tokenize(q_per_image_id[el][round]):
    #             token_image_id[el].add(t)

    train_q = []
    for q in train_q_newline:
        train_q.append(q[:-1])

    print('epoch', 'novel_qs_per_dial', 'avg_unique_qs', 'perc_games_rep', 'mutual_overlap', 'global_recall', 'local_recall_TBD')
    for epoch in range(12, 13):
        # sleep(60)

        # if epoch % 3 != 0:
        #    continue

        with open(gdse_file.format(epoch), 'r') as ff:
            gdse_dial = json.load(ff)

        # Process ReCap
        gdse_dial_new = dict()
        gdse_dial_new['data'] = []

        for id, el in enumerate(gdse_dial):
            dd = dict()
            dd['image_id'] = gdse_dial[el]['image']
            dd['flickr_url'] = gdse_dial[el]['flickr_url']
            dd['dialog'] = list()

            dial_string = gdse_dial[el]['gen_dialogue'][8:].split(' ')
            turn = dict()
            qa = ''
            for id in range(len(dial_string)):
                if '<' not in dial_string[id]:
                    qa += dial_string[id] + ' '
                else:
                    turn['question'] = qa
                    turn['answer'] = dial_string[id].replace('<', '').replace('>', '')
                    dd['dialog'].append(turn)
                    turn = dict()
                    qa = ''
            gdse_dial_new['data'].append(dd)

        recap_unique = []
        recap_repeated = 0
        recap_novel = 0
        recap_novel_set = set()
        bleu_metric_recap = 0
        recap_ent_1_list = []
        recap_ent_2_list = []
        recap_dist_1_list = []
        recap_dist_2_list = []
        recap_ent_1 = 0
        recap_ent_2 = 0
        q_tokens_recap= list()
        all_tokens_recap = set()
        recap_unique_qs = set()
        recap_q_image = dict()

        for id, el in enumerate(gdse_dial_new['data']):
            # if id in idx_above_mr:
            #     continue
            unique_q = set()
            avg_bleu_score = 0

            unigrams = []
            bigrams = []
            recap_q_image[el['image_id']] = set()

            for round in range(len(el['dialog'])):
                unique_q.add(el['dialog'][round]['question'])

                unigrams.extend(list(ngrams(nltk.word_tokenize(el['dialog'][round]['question']), 1)))
                bigrams.extend(list(ngrams(nltk.word_tokenize(el['dialog'][round]['question']), 2)))

                references = []

                for k in range(len(el['dialog'])):
                    if k != round:
                        references.append(nltk.word_tokenize(el['dialog'][k]['question']))

                q_tokens_recap.extend(nltk.word_tokenize(el['dialog'][round]['question']))

                for t in nltk.word_tokenize(el['dialog'][round]['question']):
                    all_tokens_recap.add(t)
                    recap_q_image[el['image_id']].add(t)

                recap_unique_qs.add(el['dialog'][round]['question'])

                avg_bleu_score += sentence_bleu(references, nltk.word_tokenize(el['dialog'][round]['question']))

                q = el['dialog'][round]['question']
                if str("\\") in el['dialog'][round]['question']:
                    aaa = 0

                if el['dialog'][round]['question'].lower()[:-1].replace(' ?', '?') not in train_q:
                    recap_novel += 1
                    recap_novel_set.add(el['dialog'][round]['question'])

            avg_bleu_score /= float(len(el['dialog']))
            bleu_metric_recap += avg_bleu_score
            recap_unique.append(len(unique_q))
            if len(unique_q) < len(el['dialog']):
                recap_repeated += 1
            tot_tokens = len(unigrams)
            unigram_ctr = Counter(unigrams)
            bigram_ctr = Counter(bigrams)
            cur_ent_1 = get_entropy_ctr(unigram_ctr)
            recap_ent_1 += cur_ent_1
            recap_ent_1_list.append(cur_ent_1)
            cur_ent_2 = get_entropy_ctr(bigram_ctr)
            recap_ent_2 += cur_ent_2
            recap_ent_2_list.append(cur_ent_2)

            dist_1 = len(unigram_ctr.keys()) / float(tot_tokens)
            dist_2 = len(bigram_ctr.keys()) / float(tot_tokens)

            recap_dist_1_list.append(dist_1)
            recap_dist_2_list.append(dist_2)

        novel_qs_per_dial = recap_novel/len(gdse_dial_new['data'])
        avg_unique_qs = mean(recap_unique)
        perc_games_rep = (recap_repeated / len(gdse_dial_new['data'])) * 100
        mutual_overlap = bleu_metric_recap / len(gdse_dial_new['data'])

        # inters_recap = []
        # inters_recap_content = []
        # for img_id in token_image_id:
        #     human_token = token_image_id[str(img_id)]
        #     recap_token = recap_q_image[int(img_id)]
        #     inters_recap.append(len(intersection(recap_token, human_token)) / len(human_token))
        #
        #     human_token = select_content_words(human_token, content_words)
        #     recap_token = select_content_words(recap_token, content_words)
        #     inters_recap_content.append(len(intersection(recap_token, human_token)) / len(human_token))

        # print("{} ; {} ; {} ; {} ; {}".format(epoch, novel_qs_per_dial, avg_unique_qs, perc_games_rep, mutual_overlap))
        print("{} ; {} ; {} ; {} ; {}; {}; {}".format(epoch, novel_qs_per_dial, avg_unique_qs, perc_games_rep, mutual_overlap, (len(all_tokens_recap)/len_vocab)*100, 0))
        # sleep(670)

        with open("novel_questions.txt", mode="w") as out_file:
            for x in recap_novel_set:
                out_file.write(x)
