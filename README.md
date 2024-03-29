# Reproduction project: original GuessWhat?! baseline, GDSE, RoBERTa, LXMERT

This reproduction project is part of the assignments for the Language, Vision and Interaction course by [Prof. Dr. Schlangen](https://scholar.google.com/citations?user=QoDgwZYAAAAJ&hl=en) at the University of Potsdam, part of the [Cognitive Systems Masters program](https://www.uni-potsdam.de/de/studium/studienangebot/masterstudium/master-a-z/cognitive-systems).  

We reproduced part of the experiments in the paper: Greco, C., Testoni, A., & Bernardi, R. (2020). Which Turn do Neural Models Exploit the Most to Solve GuessWhat? Diving into the Dialogue History Encoding in Transformers and LSTMs. NL4AI@AI*IA. [(link)](https://github.com/TamaraAtanasoska/dialogue-history/blob/main/project-docs/Greco%2C%20Testoni%2C%20Bernardi_2020.pdf). We focused on the blind and multimodal baseline with both LSTM and Transformer models, as well as two additional experiments where the history is reversed and the last turn is omitted. A presentation outlining the most important info from the paper and our initial plan can be found [here](https://github.com/TamaraAtanasoska/dialogue-history/blob/main/project-docs/Paper%20presentation%20%2B%20replicaition%20plan.pdf).

The team consists of [Tamara Atanasoska](https://github.com/TamaraAtanasoska), [Galina Ryazanskaya](https://github.com/flying-bear), and [Bhuvanesh Verma](https://github.com/Bhuvanesh-Verma). The team contributed to all of the tasks equally. 

If you would like to read more about the reproduction details, our comparison with the original paper, and see results from each of the base models and experiments, please read the [project report](project-docs/project-report.pdf). You can read the final report with our extended expriments that go beyond the scope of replication paper [here](project-docs/final-report.pdf). 

## Cloned repositories and code organisation

In [model-repos](model-repos/) directory, there are the [guesswhat](model-repos/guesswhat/) and the [aixia2021](model-repos/aixia2021/) cloned and edited repositories. They are clones from the [original GuessWhat repository](https://github.com/GuessWhatGame/guesswhat) and the [paper-specific ensemble repository](https://github.com/claudiogreco/aixia2021) respectively. The first one contains the model from the original GuessWhat?! [paper](https://arxiv.org/abs/1611.08481), and the second one contains a model inspired by the original baseline model and the additions that the research group contributed in order to run the experiments described in the paper we are replicating. 

We started our explorations by running the original baseline and continued to the more recent and improved repository. The code in both cloned repositories is modified to work with the version of the dataset as of June 2022, as well as some other small changes. 

## Setting up the environment and installing the required packages

In the folder ```setup/``` you can find the respective environment replication and package requirements files. 

You can either run the ```requirements.txt``` file or, if you are using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to manage your virtual environments, you can replicate the full exact environments in which we ran the code with the second pair of commands below. 

### Original Guesswhat repository
```
pip install -r setup/guesswhat/requirements.txt

OR

conda env create --name <name> --file setup/guesswhat/conda.yml python=3.6
conda activate <name>
```
### Aixia20201
```
pip install -r setup/aixia2021/requirements.txt

OR

conda env create --name <name> --file setup/aixia2021/conda.yml python=3.9
conda activate <name>
```

For the training to run faster, it is recommended to use a GPU. Our setup includes NVIDIA GeForce GTX 1080 Ti with CUDA 11.6 installed and we use the following command to install PyTorch. You can find the right Pytorch installation for your use case [here](https://pytorch.org/get-started/locally/).
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

**Important**: the only shared data between the two repositories are the dialogue jsons that are available using the first commands. The images require different naming, and the image features are generated using different models. As we preserved the project tree structure of both the projects, we expect each of the repositories to have its own data folders with all the data necessary in them.

If you decide to follow our code structure, it will look like this. The root ```data/``` folder as well as the subfolders in each model repository are populated by many generated files that appear after using the commands described in the rest of the generation docs.

```
model-repos
├── guesswhat
|   ├── config         
|   ├── out            
|   ├── data
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
    |   └── img       
    |       ├── mscoco-bottomup  #lxemert features
    |       └── raw              #all images
    |
    └── train 
```

### Dataset

The GuessWhat?! game uses two datasets: GuessWhat?! dialogues and [MS COCO](https://cocodataset.org/#home) images. These commands assume you are in either of the repositories in ```model-repos/``` already.

The three parts of the GuessWhat?! dataset(train, validation, test) can be downloaded using the following commands: 

```
wget https://florian-strub.com/guesswhat.train.jsonl.gz -P data/
wget https://florian-strub.com//guesswhat.valid.jsonl.gz -P data/
wget https://florian-strub.com//guesswhat.test.jsonl.gz -P data/
```

The MS COCO dataset can be downloaded using the first two commands. The second two will help you unzip into the ```data/img/raw``` folder. 

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

As the first step, the names of the unzipped image files need to be changed in a format that the script expects. You can do that by running the following script:

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

The only way to obtain the crop/object features is to generate them with a provided script after downloading the pre-trained VGG-16 network [slim-tensorflow](https://github.com/tensorflow/models/tree/master/research/slim). The image features can be generated by the same script. This is a computationally demanding step. **To just train the guesser**, which is the scope we cover with our reproduction, **these features are not necessary**. You would need to generate them if you would like to train the Oracle for example. 

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

**Important: these features will be created on the go as you start the training script.** There is no need to do anything and you can skip this section unless you would like to decouple the image feature creation from the training process. 

The scripts below assume that you are already in the model directory at ```model-repos/aixia2021/```.

Unlike the original repository, the Aixia2021 repository uses ResNet features. To run the models, both the image and object features will need to be generated. 

Note: the directory passed to the ```-img_dir``` option needs to contain both the train and valid images in the same directory.

#### ResNet image features
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=PYTHONPATH=PATH/TO/PROJECT/BASE/FOLDER \
python utils/ExtractImgfeatures.py \
  -image_dir data/img/raw \
  -n2n_train_set data/n2n_train_successful_data.json \
  -n2n_val_set data/n2n_val_successful_data.json \
  -image_features_json_path data/ResNet_avg_image_features2id.json \
  -image_features_path data/ResNet_avg_image_features.h5
```

#### ResNet object features
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=PYTHONPATH=PATH/TO/PROJECT/BASE/FOLDER \
python utils/extract_object_features.py \
  -image_dir data/img/raw \
  -training_set data/guesswhat.train.jsonl.gz \
  -validation_set data/guesswhat.valid.jsonl.gz \
  -objects_features_index_path data/objects_features_index_example.json \
  -objects_features_path data/objects_features_example.h5
```

#### LXMERT features

We obtained the LXMERT features files by contacting the authors. 


## Training the Guesser 

### Training the Guesser in the original GuessWhat repository

The scripts below assume that you are already in the model directory at ```model-repos/guesswhat/```. 

Add the current repository to the python path:
```
export PYTHONPATH=src:${PYTHONPATH} 
```

Create the dictionary: 
```
python src/guesswhat/preprocess_data/create_dictionary.py -data_dir data -dict_file dict.json -min_occ 3
```

In the [config/guesser/config.json](config/guesser/config.json) file, the input given to the models can be configured along with many other training parameters. You can see more information about these configurations [here](https://github.com/TamaraAtanasoska/dialogue-history/tree/main/model-repos/guesswhat/config/guesser).

To start the training run the command below. The training will print training and validation loss and error and will trigger the testing automatically at the end.

```
python src/guesswhat/train/train_guesser.py \
   -data_dir data \
   -img_dir data/img/ft_vgg_img \
   -config config/guesser/config.json \
   -exp_dir out/guesser \
   -no_thread 2 
```

### Training the four Guesser models of the Aixia20201 repository

#### General info

##### W&B integration

We have introduced [Weights & Biases](https://wandb.ai/site) as platform support to visualize and keep track of our experiments. You could take advantage of this integration by adding the option ```-exp_tracker wandb``` to the training commands. 

If you decide to use the option, Weights & Biases will ask you to log in so you can have access to the visualizations and the logging of the runs. You will be prompted to pick an option about how to use W&B, and logging in will subsequently require your W&B API key. It might be more practical for you to already finish this setup before starting the training runs with this option. You can read [here](https://docs.wandb.ai/ref/cli/wandb-login) how to do that from the command line. Creating an account before this step is necessary. 

In each of the individual training files in the [train folder](model-repos/aixia2021/train/SL) we initialise the entity and project name([example](https://github.com/TamaraAtanasoska/dialogue-history/blob/73cdd2cf8675b6297f7b1bd6bfed32589680cf7b/model-repos/aixia2021/train/SL/train_lxmert_guesser_only.py#L81). You can edit this line to add your own names, and learn more about these settings in the [W&B documentation](https://docs.wandb.ai/ref/python/init). 

##### Decider

As you will notice in the commands below, we are always training with the ```-no_decider``` flag. This is because we didn't use the [Decider component](https://github.com/TamaraAtanasoska/dialogue-history/blob/main/model-repos/aixia2021/models/Decider.py) for any of our experiments, although it is present in the repository we cloned. We haven't throughly tested either that part of the code or the impact it has on the results. You could use the Decider by omitting the ```-no_decider``` flag. Read more about the Decider [here](https://arxiv.org/pdf/1805.06960.pdf). 

#### Training

The scripts below assume that you are already in the model directory at ```model-repos/aixia2021/```.

##### Common training parameters between the models

The text below is copied from the original repo. 

```
data : Data Directory containing 
            1. ResNet image and object features,
            2. guesswhat train, val and test jsons
            3. mscoco images under img/raw sub directories
config : Config file
exp_name : Experiment Name
bin_name : Name of the trained model file
my_cpu : To select number of workers for dataloader. 
         CAUTION: If using your own system then make this True
breaking : To Break training after 5 batch, for code testing purpose
resnet : This flag will cause the program to use the image features 
         from the ResNet forward pass instead of the precomputed ones.   
modulo : This flag will cause the guesser to be updated every modulo 
         number of epochs 
no_decider : This flag will cause the decider to be turned off
num_turns : Max number of turns allowed in a dialogue
ckpt : Path to saved checkpoint
```

**Important:** For all models, the ```-test_data_dir <path>``` command line option can be added. This will trigger automatic testing with the saved checkpoint for best epoch. The saved checkpoint will always be shown at the end of training. The test directory should contain all the required files discussed in the [testing section](https://github.com/TamaraAtanasoska/dialogue-history#testing-and-running-experiments) below.

#### Language/Blind models

##### Bling LSTM model (inspired by the original GuessWhat?! model)
The original GuessWhat!? model that is featured in the original repo is part of the Aixia2021 ensemble as well. To train it, please use: 

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=PATH/TO/PROJECT/BASE/FOLDER \
python train/SL/train_lstm_guesser_only.py \
  -no_decider \
  -exp_name name \
  -bin_name name
```

##### RoBERTa

To train the model from scratch, add ```-from_scratch```.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=PATH/TO/PROJECT/BASE/FOLDER \
python train/SL/train_bert.py \
  -no_decider \
  -exp_name name \
  -bin_name name
```

#### Multimodal models

##### GDSE (V-LSTM)

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=PATH/TO/PROJECT/BASE/FOLDER \
python train/SL/train_vlstm_guesser_only.py \
  -no_decider \
  -exp_name name \
  -bin_name name
```

##### LXMERT

To train the model from scratch, add ```-from_scratch```. To use preloaded MS-COCO bottom-Up features add ```-preloaded```. If you don't train the model from scratch, you are using the pretrained version and you will need to download it. Please follow the steps below.  

```
mkdir -p lxmert/snap/pretrained 
wget http://nlp.cs.unc.edu/data/model_LXRT.pth -P lxmert/snap/pretrained
```

To train, use the following command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=PATH/TO/PROJECT/BASE/FOLDER \
python train/SL/train_lxmert_guesser_only.py \
  -no_decider \
  -exp_name name \
  -bin_name name
```

## Testing and running experiments

Although the paper we are replicating features a larger number of experiments and they are overall more granulated, we focused on exploring the effect of removing the last turn, and the effect of reversing the history. You can read more about the experiments in the [paper](https://github.com/TamaraAtanasoska/dialogue-history/blob/main/project-docs/Greco%2C%20Testoni%2C%20Bernardi_2020.pdf) discussion sections. 

The script to generate the new data json files is located at [experiments_data_prep.py](experiments_data_prep.py). It will create new json files based on the test data, for both of the experiments. 

For the original GuessWhat?! repo, the testing step is triggered automatically after the training is done. To do the testing on the experiment data, you will need to replace the ```guesswhat.test.jsonl.gz``` with the json generated by the script for the experiment you are interested in. 

To test the ```Aixia2021``` models on experiment data, you would need to take the following steps:

1. Make sure that you are already in the model directory at ```model-repos/aixia2021/```.
2. Use the [experiments_data_prep.py](experiments_data_prep.py) to prepare your data for the experiment. 
3. You can pick your folder structure, although you will need to place three files in the same folder. You will also need to copy the ```vocab.json``` file from the main data folder in your experiment subfolders. This step is important to ensure the same embeddings will be used. 

    If we pick the ```no-last-turn``` experiement as an example, the file organization could look like this: 
    ```
    data/test/experiment/no-last-turn
                         ├── guesswhat.test.jsonl.gz       #this is the output of the json modification script
                         ├── vocab.json                    #the same vocabulary file copied from the main data folder
                         └── n2n_test_successful_data.json #one of a few image feature files we will generate next 
    ```
4. Now it is time to generate the image features. The image must be regenerated every time the original dialogue json files are changed. These commands resemble the feature generation commands up, but use different scripts and different locations. 

   ```bash
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=PATH/TO/PROJECT/BASE/FOLDER/ \
    python utils/datasets/SL/prepro.py \ #use utils/datasets/SL/prepro_lxmert.py for LXMERT
    -data_dir data/test/experiment/no-last-turn \
    -data_file guesswhat.test.jsonl.gz \
    -vocab_file vocab.json \
    -split test
   ```   
    Note: the directory passed to the ```-img_dir``` option needs to contain both the train and valid images in the same directory. 
   
    ```bash
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=PATH/TO/PROJECT/BASE/FOLDER \
    python testing/ExtractTestImgfeatures.py \
    -image_dir data/img/raw \
    -n2n_test_set data/test/experiment/no-last-turn/n2n_test_successful_data.json \
    -image_features_json_path data/test/experiment/no-last-turn/ResNet_avg_image_features2id.json \
    -image_features_path data/test/experiment/no-last-turn/ResNet_avg_image_features.h5
   ```
   
    ```bash
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=PATH/TO/PROJECT/BASE/FOLDER \
    python testing/extract_test_object_features.py \
    -image_dir data/img/raw \
    -test_set data/test/experiment/no-last-turn/guesswhat.test.jsonl.gz \
    -objects_features_index_path data/test/experiment/no-last-turn/objects_features_index_example.json \
    -objects_features_path data/test/experiment/no-last-turn/objects_features_example.h5
   ```
5. Once all the image features are generated, we can run the test script for each experiment. Important to note is that you will need to know which model you would like to test. You can find your models in ```bin/SL/<model-location>```. The location of the best model is passed to the ```-best_ckpt``` option.

    ```
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=PATH/TO/PROJECT/BASE/FOLDER \
    python testing/test_lstm.py \  #test_bert instead of test_lstm for the transformer models
    -data_dir data/test/experiment/no-last-turn/ \
    -config config/SL/config_devries.json \
    -best_ckpt bin/SL/<best-model-location> \ #example model 
    -model_type blind  #or visual
    ```

## Code improvements across repositories

In the original GuessWhat repository we have improved the inconsistent documentation, adjusted the code to account for the new version of the dataset and fixed some other small errors. We contributed our changes back to the original repository making us official contributors (see [here](https://github.com/GuessWhatGame/generic/pull/1), [here](https://github.com/GuessWhatGame/guesswhat/pull/31) and [here](https://github.com/GuessWhatGame/guesswhat/pull/30)). We hope this helps people to be able to just run the repository without issues in 2022. 

The Aixia2021 repository was our main experiment repository, so we have spent more time changing and customising. We have removed all deprecation and unsafe computation warnings, and currently all models run error free. Because of our limited computational resources we spent some time on optimisation and performance improvement, resulting with up to 30% improvement of the VLSTM/GDSE model, as well as 1-2 minutes per epoch for the BERT models. You might be able to notice a more dramatic improvement if you have more powerful architecture to run them on, as for us they already ran with over 90% of the GPU usage in the first place. Some other notable changes: we removed unused dependencies like Tensorboard, introduced saving the best checkpoint and optionally triggering testing on it, as well as simplifying the image feature creation by creating them on the go and not as a separate step. For more details check the [PR history](https://github.com/TamaraAtanasoska/dialogue-history/pulls?q=is%3Apr+is%3Aclosed). 
