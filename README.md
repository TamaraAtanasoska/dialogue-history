# Reproduction project: original GuessWhat?! baseline, GDSE, RoBERTa, LXMERT

This reproduction project is part of the assignments for the Language, Vision and Interaction course by [Prof. Dr. Schlangen](https://scholar.google.com/citations?user=QoDgwZYAAAAJ&hl=en) at the University of Potsdam, part of the [Cognitive Systems Masters program](https://www.uni-potsdam.de/de/studium/studienangebot/masterstudium/master-a-z/cognitive-systems).  

We reproduced part of the expriments in the paper: Greco, C., Testoni, A., & Bernardi, R. (2020). Which Turn do Neural Models Exploit the Most to Solve GuessWhat? Diving into the Dialogue History Encoding in Transformers and LSTMs. NL4AI@AI*IA. [(link)](https://github.com/TamaraAtanasoska/dialogue-history/blob/main/project-docs/Greco%2C%20Testoni%2C%20Bernardi_2020.pdf). We focused on the blind and multimodal baseline with both LSTM and Transformer models, as well as two additional expriements where the history is reversed and the last turn is ommited. A presentation outlining the most important info from the paper and our initial plan can be found [here](https://github.com/TamaraAtanasoska/dialogue-history/blob/main/project-docs/Paper%20presentation%20%2B%20replicaition%20plan.pdf).

The team consists of: [Galina Ryazanskaya](https://github.com/flying-bear), [Bhuvanesh Verma](https://github.com/Bhuvanesh-Verma), [Tamara Atanasoska](https://github.com/TamaraAtanasoska). The team contributed to all of the tasks equally. 

## Cloned repositories and code organisation

In [model-repos](model-repos/) directory there are the [guesswhat](model-repos/guesswhat/) and the [aixia2021](model-repos/aixia2021/) cloned and edited repositories. They are clones from the [original GuessWhat repository](https://github.com/GuessWhatGame/guesswhat) and the [paper specific ensemble repository](https://github.com/claudiogreco/aixia2021) respectively. The first one contains the model from the original GuessWhat?! [paper](https://arxiv.org/abs/1611.08481), and the second one contains the original model inspired baseline and the additions that the research group contributed in order to run the experiments described in the paper we are replicating. 

We started our explorations by running the original baseline, and continued to the more recent and improved repository. The code in both cloned repositories is modified to work with the version of the dataset as of June 2022, as well as some other small changes. 

## Setting up the environment and installing the required packages

In the folder ```setup/``` you can find the respective environment replication and package requrements files. 

You can either run the ```requirements.txt``` file, or if you are using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to manage your virtual enviorments, you can replicate the full exact environments in which we ran the code with the second pair of commands below. 

### Original Guesswhat repository
```
pip install -r setup/guesswhat/requirements.txt

OR

conda env create --name <name> --file setup/guesswhat/conda.yml python=3.6
conda source activate <name>
```
### Aixia20201
```
pip install -r setup/aixia2021/requirements.txt

OR

conda env create --name <name> --file setup/aixia2021/conda.yml python=3.9
conda source activate <name>
```

In order for the training to run faster, it is recommended to use a GPU. Our setup includes NVIDIA GeForce GTX 1080 Ti with CUDA 11.6 installed and we use following command to install PyTorch. You can find the right Pytorch installation for your usecase [here](https://pytorch.org/get-started/locally/).
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

You might run into protobuf versioning errors. If so, please use the following command:
```
pip3 install --upgrade protobuf==3.20.0
```

To run the LXMERT model, the following package is necessary as well:
```
pip install git+https://github.com/bshillingford/python-sharearray
```

## Downloading the dataset and acquiring the image features

**Important**: the only shared data between the two repositories are the dialogue jsons that are available using the first commands. The images require different naming, and the image features are generated using different models. As we preserved the project tree structure of both the projects, we expect that each of the repositories have their own data folders with all the data necessary in it.

If you decide to follow our code structure, it will look like this. The root ```data/``` folder as well as the subfolders in each model repository are populated by many generated files that appear after using the commands descibed in the rest of the generation docs.

```
model-repos
├── guesswhat
|   ├── config         
|   ├── out            
|   ├── data
|   |   ├── expriments       #dialogue data for expriements
|   |   └── img      
|   |       ├── ft_vgg_img   #image features
|   |       ├── ft_vgg_crop  #crop/object image features
|   |       └── raw          #all images
|   |
|   └── src   
└── aixia20201
    ├── config         
    ├── models
    ├── utils
    ├── bin
    ├── lxmert
    ├── data  
    |   ├── expriments           #dialogue and image feature data for expriements
    |   └── img       
    |       ├── mscoco-bottomup  #lxemert features
    |       └── raw              #all images
    |
    └── train 
```

### Dataset

The GuessWhat?! game uses two datasets: GuessWhat?! dialogues and [MS COCO](https://cocodataset.org/#home) images. Before running this commands we assume you are in either of the repositories in ```model-repos/``` already.

The three parts of the GuessWhat?! dataset(train, validation, test) can be downloaded using the following commands: 

```
wget https://florian-strub.com/guesswhat.train.jsonl.gz -P data/
wget https://florian-strub.com//guesswhat.valid.jsonl.gz -P data/
wget https://florian-strub.com//guesswhat.test.jsonl.gz -P data/
```

The MS COCO dataset can be downloaded using the first two commands. The second two will help you unzip to into the ```data/img/raw``` folder. 

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

### Image features for the orginal GuessWhat model

The scripts below assume that you are already in the model directory at ```model-repos/guesswhat/```. 

As a first step, the names of the unzipped image files needs to be changed in a format that the script expects. You can do that by running the following script: 

```
python src/guesswhat/preprocess_data/rewire_coco_image_id.py \ 
   -image_dir data/img/raw \
   -data_out data/img/raw
```

Next, the VGG image and crop/object features need to be downloaded. To obtain just the image features the following download would suffice:

```
wget www.florian-strub.com/github/ft_vgg_img.zip -P data/images
unzip data/images/ft_vgg_img.zip -d data/images/
```

The only way to obtain the crop/object features is to generate them with a provided script after downloading the pre-trained VGG-16 network [slim-tensorflow](https://github.com/tensorflow/models/tree/master/research/slim). The image features can be generated by the same script. This is a computationally demanding step. **To just train the guesser**, which is the scope we cover with our reproduction, **these features are not necessary**. You would need to generate them if you would like to train the Oracle for exampple. 

```
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz -P data/
tar zxvf data/vgg_16_2016_08_28.tar.gz -C data/
```
After the download, run the script below replacing ```$mode``` with ```img``` or ```crop```.
```
python src/guesswhat/preprocess_data/extract_img_features.py \
   -img_dir data/img/raw \
   -data_dir data \
   -out_dir data/img/ft_vgg_$mode \
   -network vgg \
   -ckpt data/vgg_16.ckpt \
   -feature_name fc8 \
   -mode $mode
```

### Image features for the Aixia2021 repository

The scripts below assume that you are already in the model directory at ```model-repos/aixia2021/```.

Unlike the original repository, the Aixia2021 repository uses ResNet features. To run the models, both the image and object features will need to be generated. 

Note: the directory passed to the ```-img_dir``` option needs to contain both the train and valid images in the same directory.

#### ResNet image features
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<replace with aixia2021 path> \
python utils/ExtractImgfeatures.py \
  -image_dir data/img/raw \
  -n2n_train_set data/n2n_train_successful_data.json \
  -n2n_val_set data/n2n_val_successful_data.json \
  -image_features_json_path data/ResNet_avg_image_features2id.json \
  -image_features_path data/ResNet_avg_image_features.h5
```

#### ResNet object features
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<replace with aixia2021 path> \
python utils/extract_object_features.py \
  -image_dir data/img/raw \
  -training_set data/guesswhat.train.jsonl.gz \
  -validation_set data/guesswhat.valid.jsonl.gz \
  -objects_features_index_path data/objects_features_index_example.json \
  -objects_features_path data/objects_features_example.h5
```

#### LXMERT features

We obtained the LXMERT feature files by contacting the authors. 
