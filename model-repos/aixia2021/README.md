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


## Setup

Start by creating virtual environment and installing required packages to 
run all experiments successfully.

```bash
# change ENV_NAME to appropriate name
conda create -n ENV_NAME python=3.9
conda activate ENV_NAME
```

To run experiment faster it is recommended to use GPU. Our setup include 
NVIDIA GeForce GTX 1080 Ti with CUDA 11.6 installed and we use following 
command to install PyTorch.
```bash
# Remember to activate conda environment
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```
You can find appropriate installation for PyTorch from here: 
https://pytorch.org/get-started/locally/

Once PyTorch is installed, now we can install all other required packages 
using following command:

```bash
# Remember to activate conda environment and be in same directory where 
# requirement.txt is present
pip install -r requirement.txt
```

## Data
GuessWhat?! dataset can be downloaded using following commands:

```bash
wget https://florian-strub.com/guesswhat.train.jsonl.gz -P data/
wget https://florian-strub.com//guesswhat.valid.jsonl.gz -P data/
wget https://florian-strub.com//guesswhat.test.jsonl.gz -P data/
```
To download the MS Coco dataset, please follow the following instruction:

```bash
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip -P data/img/
unzip data/img/train2014.zip -d data/img/raw

wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip -P data/img/
unzip data/img/val2014.zip -d data/img/raw
```

## Preprocessing

To train model we require various input files mentioned in [config](config/SL/config.json) file
for respective training module. Some of these input files are not 
created on the go, so we create them separately before training.  

### ResNet image features
To get ResNet image feature, run following command:
```bash
# Here image_dir should contain both train and val images in same directory
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=PATH/TO/PROJECT/BASE/FOLDER \
python utils/ExtractImgfeatures.py \
-image_dir data/img/raw \
-n2n_train_set data/n2n_train_successful_data.json \
-n2n_val_set data/n2n_val_successful_data.json \
-image_features_json_path data/ResNet_avg_image_features2id.json \
-image_features_path data/ResNet_avg_image_features.h5
```

### ResNet object features
To get ResNet object feature, run following command:
```bash
# Here image_dir should contain both train and val images in same directory
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=PATH/TO/PROJECT/BASE/FOLDER \
python utils/extract_object_features.py \
-image_dir data/img/raw \
-training_set data/guesswhat.train.jsonl.gz \
-validation_set data/guesswhat.valid.jsonl.gz \
-objects_features_index_path data/objects_features_index_example.json \
-objects_features_path data/objects_features_example.h5
```


### LXMERT images features
TODO

## Training
Training scripts are almost same for all four models and most of the parameters
are same as well. We list common parameters below:

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
### Language Models or Blind models
1. Train LSTM model using following command:
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=PATH/TO/PROJECT/BASE/FOLDER \
python train/SL/train_lstm_guesser_only.py \
-no_decider \
-exp_name test \
-bin_name test
```

2. Train Roberta model using following command:
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=PATH/TO/PROJECT/BASE/FOLDER \
python train/SL/train_bert.py \
-no_decider \
-exp_name test \
-bin_name test
```

Use parameter `from_scratch` to train BERT model from scratch.


### Multimodal
1. Train V-LSTM
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=PATH/TO/PROJECT/BASE/FOLDER \
python train/SL/train_vlstm_guesser_only.py \
-no_decider \
-exp_name test \
-bin_name test
```

2. Train LXMERT

```bash
# Cannot be trained, required input files are not available yet
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=PATH/TO/PROJECT/BASE/FOLDER \
python train/SL/train_lxmert_guesser_only.py \
-no_decider \
-exp_name test \
-bin_name test
```
Use parameter `from_scratch` to train LXMERT model from scratch and `preloaded`
to use preloaded MS-COCO Bottom-Up features.

## Experiments (Outdated)

**NOTE**: We don't need to train model for experiments. We only manipulate test
json for respective experiment and then test them on model trained for task 
success.

In order to perform following experiments, we manipulate original GuessWhat
data using [modify_json.py](model-repos/guesswhat/src/guesswhat/preprocess_data/modify_json.py) 
script.
### No Last Turns 

For no last turn experiment, we remove last turn from each dialogue and 
create new GuessWhat json files for all splits.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=PATH/TO/PROJECT/BASE/FOLDER \
python train/SL/TRAIN_ANY_MODEL.py \
-no_decider \
-exp_name model_nlt \
-bin_name model_nlt \
-data_dir data/experiments/no-last-turns
```

### Reverse dialogue history

For reverse dialogue history experiment, we reverse dialogue history and 
create new GuessWhat json files for all splits.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=PATH/TO/PROJECT/BASE/FOLDER/ \
python train/SL/TRAIN_ANY_MODEL.py \
-no_decider \
-exp_name model_reversed \
-bin_name model_reversed \
-data_dir data/experiments/reversed-history
```

## Testing
For testing all the trained models, we need to create test files using
test data. We require following files:
1. guesswhat.test.jsonl.gz
2. vocab.json : Use vocab.json created during training or created using train data
3. n2n_test_successful_data.json
    ```bash
   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=PATH/TO/PROJECT/BASE/FOLDER/ \
   python utils/datasets/SL/prepro.py \ # Change to utils/datasets/SL/prepro_lxmert.py for Transformer based models
   -data_dir data/test \
   -data_file guesswhat.test.jsonl.gz \
   -vocab_file vocab.json \
   -split test
   ```
4. ResNet_avg_image_features.h5 and ResNet_avg_image_features2id.json
    ```bash
    # Here image_dir should contain both train and val images in same directory
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=PATH/TO/PROJECT/BASE/FOLDER \
    python testing/ExtractTestImgfeatures.py \
    -image_dir data/img/raw \
    -n2n_test_set data/test/n2n_test_successful_data.json \
    -image_features_json_path data/test/ResNet_avg_image_features2id.json \
    -image_features_path data/test/ResNet_avg_image_features.h5
   ```
5. objects_features_example.h5 and objects_features_index_example.json
    ```bash
    #Here image_dir should contain both train and val images in same directory
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=PATH/TO/PROJECT/BASE/FOLDER \
    python testing/extract_test_object_features.py \
    -image_dir data/img/raw \
    -test_set data/guesswhat.test.jsonl.gz \
    -objects_features_index_path data/test/objects_features_index_example.json \
    -objects_features_path data/test/objects_features_example.h5
   ```
   
For testing experiments, follow these same steps above to obtain data files
using the GuessWhat data for respective experiment. 

Once we have all the test files ready we can test models using best model
checkpoint saved in `bin/SL/MODEL_NAME`
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=PATH/TO/PROJECT/BASE/FOLDER \
python testing/test_lstm.py \ # test_bert for bert based model
-data_dir data/test \
-config config/SL/config_devries.json \
-best_ckpt bin/SL/blind_lstm2022_06_27_13_44/model_ensemble_blind_lstm_E_8 \
-model_type blind # or visual
```


