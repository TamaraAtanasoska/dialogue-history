# Reproduction project: original GuessWhat?! baseline, GDSE, RoBERTa, LXMERT

This reproduction project is part of the assignments for the Language, Vision and Interaction course by [Prof. Dr. Schlangen](https://scholar.google.com/citations?user=QoDgwZYAAAAJ&hl=en) at the University of Potsdam, part of the [Cognitive Systems Masters program](https://www.uni-potsdam.de/de/studium/studienangebot/masterstudium/master-a-z/cognitive-systems).  

We reproduced part of the expriments in the paper: Greco, C., Testoni, A., & Bernardi, R. (2020). Which Turn do Neural Models Exploit the Most to Solve GuessWhat? Diving into the Dialogue History Encoding in Transformers and LSTMs. NL4AI@AI*IA. [(link)](https://github.com/TamaraAtanasoska/dialogue-history/blob/main/project-docs/Greco%2C%20Testoni%2C%20Bernardi_2020.pdf). We focused on the blind and multimodal baseline with both LSTM and Transformer models, as well as two additional expriements where the history is reversed and the last turn is ommited. A presentation outlining the most important info from the paper and our initial plan can be found [here](https://github.com/TamaraAtanasoska/dialogue-history/blob/main/project-docs/Paper%20presentation%20%2B%20replicaition%20plan.pdf).

The team consists of: [Galina Ryazanskaya](https://github.com/flying-bear), [Bhuvanesh Verma](https://github.com/Bhuvanesh-Verma), [Tamara Atanasoska](https://github.com/TamaraAtanasoska). The team contributed to all of the tasks equally. 

## Cloned repositories and code organisation

In [model-repos](model-repos/) directory there are the [guesswhat](model-repos/guesswhat/) and the [aixia2021](model-repos/aixia2021/) cloned and edited repositories. They are clones from the [original GuessWhat repository](https://github.com/GuessWhatGame/guesswhat) and the [paper specific ensemble repository](https://github.com/claudiogreco/aixia2021) respectively. The first one contains the model from the original GuessWhat?! [paper](https://arxiv.org/abs/1611.08481), and the second one contains the original model inspired baseline and the additions that the research group contributed in order to run the experiments described in the paper we are replicating. 

We started our explorations by running the original baseline, and continued to the more recent and improved repository. The code in both cloned repositories is modified to work with the version of the dataset as of June 2022, as well as some other small changes. 

## Setting up the environment and installing the required packages

In the folder ```setup/``` you can find the respective environment replication and package requrements files. 

You can install all the packages listed in the requirements.txt command using:
```
pip install -r setup/<repo-of-choice>/requirements.txt
```
If you are using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to manage your virtual enviorments, than you can replicate the full exact environments in which we ran the code with the commands below.

Original Guesswhat repository: 
```
conda env create --name <name> --file setup/guesswhat/conda.yml python=3.6
conda source activate <name>
```
Aixia20201: 
```
```


## Downloading the dataset and acquiring the image features

### Dataset

The GuessWhat?! game uses two datasets: GuessWhat?! dialogues and [MS COCO](https://cocodataset.org/#home) images. 

The three parts of the GuessWhat?! dataset(train, validation, test) can be downloaded using the following commands: 

```
wget https://florian-strub.com/guesswhat.train.jsonl.gz -P data/
wget https://florian-strub.com//guesswhat.valid.jsonl.gz -P data/
wget https://florian-strub.com//guesswhat.test.jsonl.gz -P data/
```

The MS COCO dataset can be downloaded using the first two commands. The second two will help you unzip to into the ```data/img/raw``` folder. We are preserving the originally proposed folder structure, feel free to change it.  

```
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip -P data/img/
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip -P data/img/

unzip data/img/train2014.zip -d data/img/raw
unzip data/img/val2014.zip -d data/img/raw
```

If you would like if any file has been corrupted, you could use the following command:

```
md5sum $file
```

## Running the original GuessWhat model

The scripts below assume that you are already in the model directory at ```model-repos/guesswhat/```.

As a first step, the names of the unzipped image files needs to be changed in a format that the script expects. You can do that by running the following script: 

```
python src/guesswhat/preprocess_data/rewire_coco_image_id.py \ 
   -image_dir data/img/raw \
   -data_out data/img/raw
```

