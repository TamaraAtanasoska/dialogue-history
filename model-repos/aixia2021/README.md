# Grounding Dialogue History: Strengths and Weaknesses of Pre-Trained Transformers

Source code related to: Greco C., Testoni A., Bernardi R. (2021) Grounding Dialogue History: Strengths and Weaknesses of Pre-trained Transformers. In: Baldoni M., Bandini S. (eds) AIxIA 2020 – Advances in Artificial Intelligence. AIxIA 2020. Lecture Notes in Computer Science, vol 12414. Springer, Cham. https://doi.org/10.1007/978-3-030-77091-4_17.

Greco et al. (2021) is an extended version of: Greco C., Testoni A., Bernardi R. (2020) Which Turn do Neural Models Exploit the Most to Solve GuessWhat? Diving into the Dialogue History Encoding in Transformers and LSTMs. In Proceedings of the 4th Workshop on Natural Language for Artificial Intelligence (NL4AI 2020) co-located with the 19th International Conference of the Italian Association for Artificial Intelligence (AI*IA 2020), Anywhere, November 25th-27th, 2020 (pp. 29–43). [[PDF](https://raw.githubusercontent.com/claudiogreco/aixia2021/main/AIxIA_NL4AI_2020.pdf)] (errata corrige).

Please cite:
```
@InProceedings{10.1007/978-3-030-77091-4_17,
author="Greco, Claudio
and Testoni, Alberto
and Bernardi, Raffaella",
editor="Baldoni, Matteo
and Bandini, Stefania",
title="Grounding Dialogue History: Strengths and Weaknesses of Pre-trained Transformers",
booktitle="AIxIA 2020 -- Advances in Artificial Intelligence",
year="2021",
publisher="Springer International Publishing",
address="Cham",
pages="263--279",
abstract="We focus on visually grounded dialogue history encoding. We show that GuessWhat?! can be used as a ``diagnostic'' dataset to understand whether State-of-the-Art encoders manage to capture salient information in the dialogue history. We compare models across several dimensions: the architecture (Recurrent Neural Networks vs. Transformers), the input modalities (only language vs. language and vision), and the model background knowledge (trained from scratch vs. pre-trained and then fine-tuned on the downstream task). We show that pre-trained Transformers, RoBERTa and LXMERT, are able to identify the most salient information independently of the order in which the dialogue history is processed. Moreover, we find that RoBERTa handles the dialogue structure to some extent; instead LXMERT can effectively ground short dialogues, but it fails in processing longer dialogues having a more complex structure.",
isbn="978-3-030-77091-4"
}
```

## Abstract
We focus on visually grounded dialogue history encoding. We show that GuessWhat?! can be used as a “diagnostic” dataset to understand whether State-of-the-Art encoders manage to capture salient information in the dialogue history. We compare models across several dimensions: the architecture (Recurrent Neural Networks vs. Transformers), the input modalities (only language vs. language and vision), and the model background knowledge (trained from scratch vs. pre-trained and then fine-tuned on the downstream task). We show that pre-trained Transformers, RoBERTa and LXMERT, are able to identify the most salient information independently of the order in which the dialogue history is processed. Moreover, we find that RoBERTa handles the dialogue structure to some extent; instead LXMERT can effectively ground short dialogues, but it fails in processing longer dialogues having a more complex structure.
