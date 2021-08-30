# Action Prediction for Conversational Agents
This work focuses on building a BERT model to predict actions and attributes as requested
by the Sub-Task#1 of the [Situated Interactive MultiModal Conversations (SIMMC) Challenge 2020](https://github.com/facebookresearch/simmc).

The project is made up of one package, `ap4ca`,  with two subpackages, `model` and `simmc`.

## Module `ap4ca`
This package contains the module `run.py`, the callable script that runs the model building, training and evaluation and that stores the results 
of the execution.
To see how to run the model and the description of the arguments accepted, go to the section [How to run](#how-to-run).

### Directory `results`
In this directory are stored the output of the runs we executed, a notebook for analyzing it and the plots of the analysis.

## Subpackage `simmc`
The subpackage contains the module `action_evaluation.py` from the [SIMMC](https://github.com/facebookresearch/simmc) project 
used for compute the metrics for the model evaluation and the module `utils.py` with some utility functions.

All the contents of this package related to the [SIMMC](https://github.com/facebookresearch/simmc) project are taken from
the cited github repository from the branch `master`, commit `cc93037eaca038ea3993cbe3f87bd032d38c3091`.

### Module `action_evaluation.py`
From this module the package uses the `evaluate_action_prediction` to compute the metrics for the evaluation of
the goodness of the model.

### Module `utils.py`
The module exposes the function `get_data`. It parses the SIMMC dataset and returns the content as a `pandas dataframe`, 
if the function is called with the `format` argument set to "df" (the default value), otherwise it returns the parsed json.
When the function is called to build the dataframe, it manipulates the input to insert the transcripts from the previous
turn inside the same dialog.

### Directory `data`
The directory contains the dataset for the project divided in three `json` files for the training (`fashion_train_dials_api_calls.json`), 
the validation (`fashion_dev_dials_api_calls.json`) and the test (`fashion_devtest_dials_api_calls.json`) step.
The datasets include multimodal context of the items appearing in each scene, and contextual NLU (natural language under-standing), 
NLG (natural language generation) and coreference annotations using a novel and unified framework of SIMMC conversational 
acts for both user and assistant utterances.

The files are the output of the script `extract_actions_fashion.py` given by the [SIMMC](https://github.com/facebookresearch/simmc) project.

## Subpackage `model`
The model package split in two classes, `Runner` from the `runner` module and `CustomBERTModel` from the `model` module,
and functions utilities from `loaders` and `utils` modules.

### Module `loaders`
The function `get_dataloader` returns a `torch Dataloader`.
The `runner` uses it to build the three dataloader needed for the computation.

### Module `model`
The model exports a class that is an extension of the torch model and that owns a BertModel built form a pretrained one (the type
can be passed via arguments, as explained in the [How To](#how-to-run) section).
The constructor accepts also two other arguments defining the dimension of the model output layer.

### Module `runner`
The class `Runner` is the core of the package and manages the full cycle of the classifier.
It builds the model depending on the arguments passed as described in the [Hot to Run](#how-to-run) section.
The constructor builds the model and prepares the execution storing dataloaders, tensors with encoded transcript, and so on.
The training loop and the validation can be called, respectively, with the `train_and_eval` and `validate` functions.
Finally, the `store_results` function saves the pandas dataframe as pickle files.

### Module `utils`
The `utils.py` module has two functions, `get_device` to retrieve the processor unit for the computations, and `loss_fn`
that returns the loss function that combines the CrossEntropyLoss computed for the actions labels and the BCELoss for
the attributes labels.

## Installation
The project requires the **version 3** of python.
Furthermore, to run the project, you need to install the following dependencies (e.g. with pip command):
```bash
pip install transformers==4.9.2
pip install absl-py
pip install pandas
pip install seaborn
pip install matplotlib
pip install numpy
pip install datetime
pip install torch==1.9.0
pip install sklearn
```

## How to run
The `run.py` script is callable from the directory containing the package with the command:
```bash
python ap4ca/run.py
```
### Command line options

`seed` Set for deterministic behaviour. Default: a random integer in the range 10, 1000000.

`results_path` Path for saving results. Default: `ap4ca/results`.

`pretrained_set` Pretrained bert model to be used. Default: `bert-base-uncased`.

`use_next` Concatenate with previous transcript. Pass `--use_next` for `True` (default) or `--nouse_next` for `False`.
If the value is `True` the model concatenate the transcript with the action to be predicted with the user and the system 
transcripts from the preceding turn of the same dialog.

`batch_size` Dimension of loaders batches. Default: 12.

`epochs` Number of epochs for training loop. Default: 6.

`hidden_output_dim` Number of node for the last hidden layer added after the BERT layers. Default: 768.

`learning_rate` The learning rate used by the optimizer. Default: 5e-5.

`tolerance` The tolerance (epsilon) used by the optimizer. Default: 1e-8.

`action_activation` Activation function for **actions** classification. Currently, are accepted only two functions: 
`softmax` (the default) and `sigmoid`.

`device_type` Device type for tensor computing (GPU or TPU). Currently, is accepted only `GPU`, that is the default.

`training_data` Full path of the dataset for the training step. Default: `ap4ca/simmc/data/fashion_train_dials_api_calls.json` 

`validation_data` Full path of the dataset for the validation. Default: `ap4ca/simmc/data/fashion_devtest_dials_api_calls.json`.

`test_data` Full path of the dataset for the test step. Default: `ap4ca/simmc/data/fashion_devtest_dials_api_calls.json`

`w_AddToCart` Cross Entropy Loss weight for 'AddToCart' action class. Default: 1.0.

`w_None` Cross Entropy Loss weight for 'None' action class. Default: 1.0.

`w_SearchDatabase` Cross Entropy Loss weight for 'SearchDatabase' action class. Default: 1.0.

`w_SearchMemory` Cross Entropy Loss weight for 'SearchMemory' action class. Default: 1.0.

`w_SpecifyInfo` Cross Entropy Loss weight for 'SpecifyInfo' action class. Default: 1.0.

`num_of_samples` Number of samples to be generated. If the number is great than one, the possible `seed` argument is 
ignored to guarantee that every sample is different from the other generated. Default: 1.

## TODO
- Implementation for TPU computing

## Contact


## License
This package is released under the MIT License

