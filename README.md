# Reproduction project: Original GuessWhat?! baseline, GDSE, RoBERTa, LXMBERT

This reproduction project is part of the assignments for the Language, Vision and Interaction course by [Prof. Dr. Schlangen](https://scholar.google.com/citations?user=QoDgwZYAAAAJ&hl=en) at the University of Potsdam, part of the [Cognitive Systems Masters program](https://www.uni-potsdam.de/de/studium/studienangebot/masterstudium/master-a-z/cognitive-systems).  

We reproduced part of the expriments in the paper: Greco, C., Testoni, A., & Bernardi, R. (2020). Which Turn do Neural Models Exploit the Most to Solve GuessWhat? Diving into the Dialogue History Encoding in Transformers and LSTMs. NL4AI@AI*IA. [(link)](http://ceur-ws.org/Vol-2735/paper31.pdf). We focused on the blind and multimodal baseline with both LSTM and Transformer models, as well as two additional expriements where the history is reversed and the last turn is ommited. 

The team consists of: [Galina Ryazanskaya](https://github.com/flying-bear), [Bhuvanesh Verma](https://github.com/Bhuvanesh-Verma), [Tamara Atanasoska](https://github.com/TamaraAtanasoska). The team contributed to all of the tasks equally. 

## Cloned repositories and code organisation

In [model-repos](model-repos/) directory there are the [guesswhat](model-repos/guesswhat/) and the [aixia2021](model-repos/aixia2021/) cloned and edited repositories. They are clones from the [original GuessWhat repository](https://github.com/GuessWhatGame/guesswhat) and the [paper specific ensemble repository](https://github.com/claudiogreco/aixia2021) respectively. The first one contains the model from the original GuessWhat?! [paper](https://arxiv.org/abs/1611.08481), and the second one contains the original model inspired baseline and the additions that the research group contributed in order to run the experiments described in the paper. 

We started our explorations by running the original baseline, and continued to the more recent and improved repository. The code in both cloned repositories is modified to work with the version of the dataset as of June 2022, as well as some other small changes. 

## Downloading the dataset and image features
