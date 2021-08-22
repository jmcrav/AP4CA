# Script to build samples exploiting TPUs on google cloud platform
import random
import torch
from torch import nn
import torch_xla
import torch_xla.core.xla_model as xm
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
import json
import pandas as pd
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import time
import datetime
import action_evaluation as evaluation
import tensorflow as tf

TPUdev = xm.xla_device()

# Model parameters
exec_params = {
    'batch': 12,
    'epochs': 6,
    'hidden_output_dim': 768,
    'seed': random.randint(100,10000000),
    'learning_rate': 5e-5,
    'tolerance': 1e-8,
    'action_activation': 'softmax'
}

# Single or two consecutive transcript for each set item
use_next = True

# Dataframes builder
def createDataframe(json_file):
    with open(json_file) as f:
        dictftdac = json.load(f)

    data = []

    for e in dictftdac:
        dialog_id = e['dialog_id']
        actions = e['actions']
        focus_images = e['focus_images']

        for a in actions:

            turn_idx = a['turn_idx']
            action = a['action']
            action_supervision = a['action_supervision']
            transcript = a['transcript']
            transcript_annotated = a['transcript_annotated']
            system_transcript = a['system_transcript']
            system_transcript_annotated = a['system_transcript_annotated']

            row = {
                "dialog_id": dialog_id,
                'turn_idx': turn_idx,
                'action': action,
                'action_supervision': action_supervision,
                'focus_images': focus_images,
                'transcript': transcript,
                'transcript_annotated': transcript_annotated,
                'system_transcript': system_transcript,
                'system_transcript_annotated': system_transcript_annotated,
                'previous_transcript': "",
                'previous_system_transcript': ""
            }
            if (action_supervision != None):
                if 'focus' in action_supervision:
                    acsf = {'focus': action_supervision['focus']}
                else:
                    acsf = {'focus': None}

                if 'attributes' in action_supervision:
                    acsa = {'attributes': action_supervision['attributes']}
                else:
                    acsa = {'attributes': []}
            else:
                acsf = {'focus': None}
                acsa = {'attributes': []}

            row.update(acsf)
            row.update(acsa)

            data.append(row)

    # Conservo id turno e risposta sistema per provare a implementare una soluzione articolata
    df = pd.DataFrame(data, columns=['dialog_id', 'turn_idx', 'transcript', 'action', 'attributes', 'system_transcript',
                                     'transcript_annotated', 'system_transcript_annotated', 'previous_transcript',
                                     'previous_system_transcript'])

    return df


class CustomBERTModel(nn.Module):

    def __init__(self):

        super(CustomBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        ### New layers:
        self.linear_intermedio = nn.Linear(768, exec_params['hidden_output_dim'])
        # provare ad aggiungere ulteriori layer intermedi per ridurre le dimensioni fino ad arrivare all'output richiesto
        self.linear_actions = nn.Linear(exec_params['hidden_output_dim'], 5)
        self.linear_attributes = nn.Linear(exec_params['hidden_output_dim'], len(mlb.classes_))  # num attributi?

    def forward(self, ids, mask):
        # controllare che l'output non rappresenti solo lo stato interno dovuto al token CLS
        output = self.bert(ids, attention_mask=mask)
        # print(f"Type output{type(output)}")
        # for p in output:
        #   print(p)
        #   print(type(output[p]))
        #   print(output[p])

        # prendiamo il campo last_hidden_state dall'oggetto output; last hidden state rappresenta il tensore
        # in uscita dallo step di forward del BertModel
        last_hidden_state_output = output["last_hidden_state"]
        # last_hidden_state has the following shape: (batch_size, sequence_length, 768)
        # stiamo passando solo il token CLS ai layer successivi
        linear_output_intermedio = self.linear_intermedio(last_hidden_state_output[:, 0, :].view(-1, 768))
        # linear_output_intermedio = self.linear_intermedio(pooled_output)

        linear_output_actions = self.linear_actions(linear_output_intermedio)
        # linear_output_actions = self.sftmx(linear_output_actions)
        # linear_output_actions = nn.functional.softmax(linear_output_actions)
        # Test sigmoid for increasing perplexity performance
        if (exec_params['action_activation'] == 'softmax'):
            linear_output_actions = nn.functional.softmax(linear_output_actions, dim=1)
        else:
            linear_output_actions = torch.sigmoid(linear_output_actions)
        # linear_output_actions = nn.functional.relu(linear_output_actions)
        linear_output_attributes = self.linear_attributes(linear_output_intermedio)
        # linear_output_attributes = self.sig(linear_output_attributes)
        linear_output_attributes = torch.sigmoid(linear_output_attributes)

        return {'actions': linear_output_actions, 'attributes': linear_output_attributes}

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy_actions(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return {'matched': np.sum(pred_flat == labels_flat), 'counts': len(labels_flat)}

def flat_accuracy_attributes(preds, labels):
    tot_preds = preds.shape[0]
    preds_int = np.rint(preds)
    tot_eq = 0
    for i in range(tot_preds):
        comparison = preds_int[i] == labels[i]
        if comparison.all():
            tot_eq += 1
    return {'matched': tot_eq, 'counts' : tot_preds}


def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# Loss function definition
def MyBERT_loss(logits, actions_labels, attributes_labels):
    actions_logits = logits['actions'].to(TPUdev)
    attributes_logits = logits['attributes'].to(TPUdev)
    loss_actions_fn = nn.CrossEntropyLoss()
    loss_attributes_fn = nn.BCELoss()
    loss_actions = loss_actions_fn(actions_logits, actions_labels)
    loss_attributes = loss_attributes_fn(attributes_logits, attributes_labels.float())
    return loss_actions + loss_attributes


# Building training, validation and test dataframe
df_training = createDataframe('./extr_output/fashion_train_dials_api_calls.json')
df_validation = createDataframe('./extr_output/fashion_dev_dials_api_calls.json')
df_test = createDataframe('./extr_output/fashion_devtest_dials_api_calls.json')

# Add previous transcript column for convenience in the datasets
# TODO list map or comprhension
#Training
df_training.sort_values(by=['dialog_id', 'turn_idx'])
for i in range(1,(len(df_training))):
    if(i<(len(df_training)) and  df_training['dialog_id'][i] == df_training['dialog_id'][i-1]):
        df_training.loc[i,'previous_transcript'] = df_training['transcript'][i-1]
        df_training.loc[i,'previous_system_transcript'] = df_training['system_transcript'][i-1]

#Validation
df_validation.sort_values(by=['dialog_id', 'turn_idx'])
for i in range(1,(len(df_validation))):
    if(i<(len(df_validation)) and  df_validation['dialog_id'][i] == df_validation['dialog_id'][i-1]):
        df_validation.loc[i,'previous_transcript'] = df_validation['transcript'][i-1]
        df_validation.loc[i,'previous_system_transcript'] = df_validation['system_transcript'][i-1]

#Evaluation
df_test.sort_values(by=['dialog_id', 'turn_idx'])
for i in range(1,(len(df_test))):
    if(i<(len(df_test)) and  df_test['dialog_id'][i] == df_test['dialog_id'][i-1]):
        df_test.loc[i,'previous_transcript'] = df_test['transcript'][i-1]
        df_test.loc[i,'previous_system_transcript'] = df_test['system_transcript'][i-1]

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
# Training
transcripts_tr = df_training.transcript.values
previous_transcript_tr = df_training.previous_transcript.values
previous_system_transcript_tr = df_training.previous_system_transcript.values
action_labels_tr = df_training.action.values
attributes_labels_tr=df_training.attributes.values

# Validation
transcripts_vd = df_validation.transcript.values
previous_transcript_vd = df_validation.previous_transcript.values
previous_system_transcript_vd = df_validation.previous_system_transcript.values
action_labels_vd = df_validation.action.values
attributes_labels_vd=df_validation.attributes.values
dialog_ids_vd = df_validation.dialog_id.values
turn_idxs_vd = df_validation.turn_idx.values

# Evaluation
transcripts_tst = df_test.transcript.values
previous_transcript_tst = df_test.previous_transcript.values
previous_system_transcript_tst = df_test.previous_system_transcript.values
action_labels_tst = df_test.action.values
attributes_labels_tst=df_test.attributes.values
dialog_ids_tst = df_test.dialog_id.values
turn_idxs_tst = df_test.turn_idx.values

# Compute max tensors dimension
# Training
max_len_tr = 0

# For every sentence...
for i in range(0, len(transcripts_tr)):

    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.

    if (previous_transcript_tr[i] != "" and use_next):
        input_ids = tokenizer.encode(previous_transcript_tr[i] + " " + previous_system_transcript_tr[i],
                                     transcripts_tr[i], add_special_tokens=True)
    else:
        input_ids = tokenizer.encode(transcripts_tr[i], add_special_tokens=True)

    # Update the maximum sentence length.
    max_len_tr = max(max_len_tr, len(input_ids))

# Validation
max_len_vd = 0

# For every sentence...
for i in range(0, len(transcripts_vd)):

    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    if (previous_transcript_vd[i] != "" and use_next):
        input_ids = tokenizer.encode(previous_transcript_vd[i] + " " + previous_system_transcript_vd[i],
                                     transcripts_vd[i], add_special_tokens=True)
    else:
        input_ids = tokenizer.encode(transcripts_vd[i], add_special_tokens=True)

    # Update the maximum sentence length.
    max_len_vd = max(max_len_vd, len(input_ids))

# Test
max_len_tst = 0

for i in range(0,len(transcripts_tst)):
    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    if (previous_transcript_tst[i] != "" and use_next):
        input_ids = tokenizer.encode(previous_transcript_tst[i]+ " " + previous_system_transcript_tst[i],transcripts_tst[i], add_special_tokens=True)
    else:
        input_ids = tokenizer.encode(transcripts_tst[i], add_special_tokens=True)

    # Update the maximum sentence length.
    max_len_tst = max(max_len_tst, len(input_ids))

max_len = max(max_len_tr, max_len_vd, max_len_tst)

# Label encoding
mlb = MultiLabelBinarizer()
attributes_labels_all = np.concatenate((attributes_labels_tr, attributes_labels_vd,attributes_labels_tst), axis=None)
attr_yt = mlb.fit_transform(attributes_labels_all)
attributes_labels_tr_vect = attr_yt[0:len(attributes_labels_tr)]
attributes_labels_vd_vect = attr_yt[len(attributes_labels_tr):(len(attributes_labels_tr)+len(attributes_labels_vd))]
attributes_labels_tst_vect = attr_yt[(len(attributes_labels_tr)+len(attributes_labels_vd)):]

# Set torch seed for deterministic behaviour
torch.manual_seed(exec_params['seed'])

# Tokenization
le = preprocessing.LabelEncoder()

# Tokenize Train Data
action_labels_encoded_tr = le.fit_transform(action_labels_tr)

input_ids_tr = []
attention_masks_tr = []
# For every sentence...
for i in range(0, len(df_training)):
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.

    if (previous_transcript_tr[i] != "" and use_next):
        encoded_dict = tokenizer.encode_plus(
            previous_transcript_tr[i] + " " + previous_system_transcript_tr[i],  # Sentence to encode.
            transcripts_tr[i],  # next sentece to encode
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            truncation=True,
            max_length=max_len,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
    else:
        encoded_dict = tokenizer.encode_plus(
            transcripts_tr[i],  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            truncation=True,
            max_length=max_len,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

    # Add the encoded sentence to the list.
    input_ids_tr.append(encoded_dict['input_ids'])

    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks_tr.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids_tr = torch.cat(input_ids_tr, dim=0)
attention_masks_tr = torch.cat(attention_masks_tr, dim=0)
labels_actions_tr = torch.tensor(action_labels_encoded_tr, device=TPUdev)
labels_attributes_tr = torch.tensor(attributes_labels_tr_vect, device=TPUdev)

# Tokenize Validation Data
action_labels_encoded_vd = le.fit_transform(action_labels_vd)

input_ids_vd = []
attention_masks_vd = []
# For every sentence...
for i in range(0, len(df_validation)):
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.

    if (previous_transcript_vd[i] != "" and use_next):
        encoded_dict = tokenizer.encode_plus(
            previous_transcript_vd[i] + " " + previous_system_transcript_vd[i],  # Sentence to encode.
            transcripts_vd[i],  # next sentece to encode
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            truncation=True,
            max_length=max_len,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
    else:
        encoded_dict = tokenizer.encode_plus(
            transcripts_vd[i],  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            truncation=True,
            max_length=max_len,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

    # Add the encoded sentence to the list.
    input_ids_vd.append(encoded_dict['input_ids'])

    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks_vd.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids_vd = torch.cat(input_ids_vd, dim=0)
attention_masks_vd = torch.cat(attention_masks_vd, dim=0)
labels_actions_vd = torch.tensor(action_labels_encoded_vd, device=TPUdev)
labels_attributes_vd = torch.tensor(attributes_labels_vd_vect, device=TPUdev)
dialog_ids_vd = torch.tensor(dialog_ids_vd, device=TPUdev)
turn_idxs_vd = torch.tensor(turn_idxs_vd, device=TPUdev)

# Tokenize Evaluation Data
action_labels_encoded_tst = le.fit_transform(action_labels_tst)

input_ids_tst = []
attention_masks_tst = []
# For every sentence...
for i in range(0, len(df_test)):
    # for t in transcripts_tst:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.

    # Aggiungere "and False" PER UTILIZZARE sempre la tokenizzazione senza concatenazione
    if (previous_transcript_tst[i] != "" and use_next):
        encoded_dict = tokenizer.encode_plus(
            previous_transcript_tst[i] + " " + previous_system_transcript_tst[i],  # Sentence to encode.
            transcripts_tst[i],  # next sentece to encode
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            truncation=True,
            max_length=max_len,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
    else:
        encoded_dict = tokenizer.encode_plus(
            transcripts_tst[i],  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            truncation=True,
            max_length=max_len,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

    # Add the encoded sentence to the list.
    input_ids_tst.append(encoded_dict['input_ids'])

    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks_tst.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids_tst = torch.cat(input_ids_tst, dim=0)
attention_masks_tst = torch.cat(attention_masks_tst, dim=0)
labels_actions_tst = torch.tensor(action_labels_encoded_tst, device=TPUdev)
labels_attributes_tst = torch.tensor(attributes_labels_tst_vect, device=TPUdev)
dialog_ids_tst = torch.tensor(dialog_ids_tst, device=TPUdev)
turn_idxs_tst = torch.tensor(turn_idxs_tst, device=TPUdev)


# TRAINING
train_dataset = TensorDataset(input_ids_tr, attention_masks_tr, labels_actions_tr, labels_attributes_tr)
val_dataset = TensorDataset(input_ids_vd, attention_masks_vd, labels_actions_vd, labels_attributes_vd, dialog_ids_vd, turn_idxs_vd)
tst_dataset = TensorDataset(input_ids_tst, attention_masks_tst, labels_actions_tst, labels_attributes_tst, dialog_ids_tst, turn_idxs_tst)

# The DataLoader needs to know our batch size for training, so we specify it
# here. For fine-tuning BERT on a specific task, the authors recommend a batch
# size of 16 or 32.
# With size 32 GeForce RTX 2060 with 6GB run out of memory
batch_size = exec_params['batch']

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order.

train_dataloader = DataLoader(
    train_dataset,  # The training samples.
    sampler = RandomSampler(train_dataset), # Select batches randomly
    batch_size = batch_size # Trains with this batch size.
)

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
    val_dataset, # The validation samples.
    sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
    batch_size = batch_size # Evaluate with this batch size.
)

#ho controllato nel colab su cui ci basiamo, anche lui usa un Sequential Sampler per il dataset di evaluation
evaluation_dataloader = DataLoader(
    tst_dataset, # The validation samples.
    sampler = SequentialSampler(tst_dataset), # Pull out batches sequentially.
    batch_size = batch_size # Evaluate with this batch size.
)

# Build BERT model
model = CustomBERTModel()
optimizer = AdamW(model.parameters(),
                  lr = exec_params['learning_rate'], # args.learning_rate - default is 5e-5
                  eps = exec_params['tolerance'] # args.adam_epsilon  - default is 1e-8.
                  )

# Training loop
epochs = exec_params['epochs']
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

with open('./extr_output/fashion_dev_dials_api_calls.json') as f:
    dev_dials = json.load(f)

# Set the seed value all over the place to make this reproducible.
seed_val = exec_params['seed']

random.seed(seed_val)
np.random.seed(seed_val)

# We'll store a number of quantities such as training and validation loss,
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

test_batch = []

# For each epoch...
for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode. Don't be mislead--the call to
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train().to(TPUdev)

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 400 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: actions labels
        #   [3]: attributes labels
        b_input_ids = batch[0].to(TPUdev)
        b_input_mask = batch[1].to(TPUdev)
        b_labels_actions = batch[2].to(TPUdev)
        b_labels_attributes = batch[3].to(TPUdev)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because
        # accumulating the gradients is "convenient while training RNNs".
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # In PyTorch, calling `model` will in turn call the model's `forward`
        # function and pass down the arguments. The `forward` function is
        # documented here:
        # https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
        # The results are returned in a results object, documented here:
        # https://huggingface.co/transformers/main_classes/output.html#transformers.modeling_outputs.SequenceClassifierOutput
        # Specifically, we'll get the loss (because we provided labels) and the
        # "logits"--the model outputs prior to activation.
        result = model(b_input_ids,
                       mask=b_input_mask)

        loss = MyBERT_loss(result, b_labels_actions, b_labels_attributes)

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value
        # from the tensor.
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.from transformers import BertModel, BertConfig
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    print(f"End of epoch {epoch_i}")

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.mlb.inverse_transform(attr_yt[3].reshape(1, -1))
    model.eval()

    # Tracking variables
    total_eval_accuracy_classification = {'matched': 0, 'counts': 0}
    total_eval_accuracy_multilabel = {'matched': 0, 'counts': 0}
    total_eval_loss = 0
    nb_eval_steps = 0

    batch_number = 0

    # Dictionary for action_evaluation
    model_actions = {}

    # Evaluate data for one epoch
    for batch in validation_dataloader:

        batch_number += 1

        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(TPUdev)
        b_input_mask = batch[1].to(TPUdev)
        b_labels_actions = batch[2].to(TPUdev)
        b_labels_attributes = batch[3].to(TPUdev)
        b_dialog_ids = batch[4].to(TPUdev)
        b_turn_idxs = batch[5].to(TPUdev)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            result = model(b_input_ids,
                           mask=b_input_mask)

        # Get the loss and "logits" output by the model. The "logits" are the
        # output values prior to applying an activation function like the
        # softmax.
        loss = MyBERT_loss(result, b_labels_actions, b_labels_attributes)

        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        # logits = logits.detach().cpu().numpy()
        # label_ids = b_labels.to('cpu').numpy()

        actions_logits_foracc = result['actions'].detach().cpu().numpy()
        attributes_logits_foracc = result['attributes'].detach().cpu().numpy()
        actions_labels_foracc = b_labels_actions.to('cpu').numpy()
        attributes_labels_foracc = b_labels_attributes.to('cpu').numpy()

        # TODO: definire la nostra funzione di accuracy

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        accuracy_classification = flat_accuracy_actions(actions_logits_foracc, actions_labels_foracc)
        accuracy_multilabel = flat_accuracy_attributes(attributes_logits_foracc, attributes_labels_foracc)

        total_eval_accuracy_classification['matched'] += accuracy_classification['matched']
        total_eval_accuracy_classification['counts'] += accuracy_classification['counts']
        total_eval_accuracy_multilabel['matched'] += accuracy_multilabel['matched']
        total_eval_accuracy_multilabel['counts'] += accuracy_multilabel['counts']
        # Salvo dati elaborazione batch per debug/analisi
        test_batch.append({
            'epoch': epoch_i + 1,
            'batchnum': batch_number,
            'actions_logits': actions_logits_foracc,
            'actions_labels': actions_labels_foracc,
            'attributes_logits': attributes_logits_foracc,
            'attributes_labels': attributes_labels_foracc,
            'accuracy_classification': accuracy_classification,
            'accuracy_multilabel': accuracy_multilabel,
        })

        # Fill dictionary for action_evaluation
        for el_i in range(len(actions_logits_foracc)):
            dialog_id = b_dialog_ids[el_i]
            action_log_prob = {}
            for act_i in range(len(actions_logits_foracc[el_i])):
                # todo: controllare che la probabilità predetta sia in scala logaritmica (?? potrebbe essere fonte di errori)
                action_log_prob[le.classes_[act_i]] = np.log(actions_logits_foracc[el_i][act_i])
            # attributes = {}
            attributes = []
            # attributes_list = np.rint(attributes_logits_foracc[el_i])
            attributes_list = np.array(attributes_logits_foracc[el_i])
            for attr in range(len(attributes_list)):
                attribute = mlb.classes_[attr]
                # attributes[mlb.classes_[attr]] = attributes_list[attr]
                if attributes_list[attr] >= 0.5:
                    attributes.append(attribute)
            prediction = {
                'action': le.classes_[np.argmax(actions_logits_foracc[el_i])],
                'action_log_prob': action_log_prob,
                'attributes': {'attributes': attributes},
                'turn_id': b_turn_idxs[el_i]
            }
            if dialog_id in model_actions:
                model_actions[dialog_id]['predictions'].append(prediction)
            else:
                predictions = list()
                predictions.append(prediction)
                model_actions[dialog_id] = {
                    'dialog_id': dialog_id,
                    'predictions': predictions
                }

    # Report the final accuracy for this validation

    # avg_val_accuracy_classification = total_eval_accuracy_classification / len(validation_dataloader)
    # avg_val_accuracy_multilabel = total_eval_accuracy_multilabel / len(validation_dataloader)
    avg_val_accuracy_classification = total_eval_accuracy_classification['matched'] / \
                                      total_eval_accuracy_classification['counts']
    avg_val_accuracy_multilabel = total_eval_accuracy_multilabel['matched'] / total_eval_accuracy_multilabel['counts']
    print("  Accuracy for classification (actions): {0:.4f}".format(avg_val_accuracy_classification))
    print("  Accuracy for multilabel-classification (attributes): {0:.4f}".format(avg_val_accuracy_multilabel))

    # Reference implementation: evaluation of action prediction along with attributes
    metrics = evaluation.evaluate_action_prediction(dev_dials, model_actions.values())
    # print("model_actions passed to the evaluator:")
    # for v in model_actions.values():
    #   print(v)
    print("***************************************")
    print("Reference evaluation metrics:")
    print(metrics)

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.4f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur. class.': avg_val_accuracy_classification,
            'Valid. Accur. mult.label': avg_val_accuracy_multilabel,
            'Training Time': training_time,
            'Validation Time': validation_time,
            'metrics': metrics
        }
    )

# Evaluation on Test Set
with open('./extr_output/fashion_devtest_dials_api_calls.json') as f:
    devtest_dials = json.load(f)

# Tracking variables
total_eval_accuracy_classification = {'matched': 0, 'counts': 0}
total_eval_accuracy_multilabel = {'matched': 0, 'counts': 0}

model_actions = {}
# Put model in evaluation mode
model.eval()

for batch in evaluation_dataloader:

    # Unpack this training batch from our dataloader.
    #
    # As we unpack the batch, we'll also copy each tensor to the GPU using
    # the `to` method.
    #
    # `batch` contains three pytorch tensors:
    #   [0]: input ids
    #   [1]: attention masks
    #   [2]: labels
    b_input_ids = batch[0].to(TPUdev)
    b_input_mask = batch[1].to(TPUdev)
    b_labels_actions = batch[2].to(TPUdev)
    b_labels_attributes = batch[3].to(TPUdev)
    b_dialog_ids = batch[4].to(TPUdev)
    b_turn_idxs = batch[5].to(TPUdev)

    # Tell pytorch not to bother with constructing the compute graph during
    # the forward pass, since this is only needed for backprop (training).
    with torch.no_grad():
        # Forward pass, calculate logit predictions.
        # token_type_ids is the same as the "segment ids", which
        # differentiates sentence 1 and 2 in 2-sentence tasks.
        result = model(b_input_ids, mask=b_input_mask)

    actions_logits_foracc = result['actions'].detach().cpu().numpy()
    attributes_logits_foracc = result['attributes'].detach().cpu().numpy()
    actions_labels_foracc = b_labels_actions.to('cpu').numpy()
    attributes_labels_foracc = b_labels_attributes.to('cpu').numpy()

    # Calculate the accuracy for this batch of test sentences, and
    # accumulate it over all batches.
    accuracy_classification = flat_accuracy_actions(actions_logits_foracc, actions_labels_foracc)
    accuracy_multilabel = flat_accuracy_attributes(attributes_logits_foracc, attributes_labels_foracc)

    total_eval_accuracy_classification['matched'] += accuracy_classification['matched']
    total_eval_accuracy_classification['counts'] += accuracy_classification['counts']
    total_eval_accuracy_multilabel['matched'] += accuracy_multilabel['matched']
    total_eval_accuracy_multilabel['counts'] += accuracy_multilabel['counts']

    # Fill dictionary for action_evaluation
    for el_i in range(len(actions_logits_foracc)):
        dialog_id = b_dialog_ids[el_i]
        action_log_prob = {}
        for act_i in range(len(actions_logits_foracc[el_i])):
            # todo: controllare che la probabilità predetta sia in scala logaritmica (?? potrebbe essere fonte di errori)
            action_log_prob[le.classes_[act_i]] = np.log(actions_logits_foracc[el_i][act_i])
        # attributes = {}
        attributes = []
        # attributes_list = np.rint(attributes_logits_foracc[el_i])
        attributes_list = np.array(attributes_logits_foracc[el_i])
        for attr in range(len(attributes_list)):
            attribute = mlb.classes_[attr]
            # attributes[mlb.classes_[attr]] = attributes_list[attr]
            if attributes_list[attr] >= 0.5:
                attributes.append(attribute)
        prediction = {
            'action': le.classes_[np.argmax(actions_logits_foracc[el_i])],
            'action_log_prob': action_log_prob,
            'attributes': {'attributes': attributes},
            'turn_id': b_turn_idxs[el_i]
        }
        if dialog_id in model_actions:
            model_actions[dialog_id]['predictions'].append(prediction)
        else:
            predictions = list()
            predictions.append(prediction)
            model_actions[dialog_id] = {
                'dialog_id': dialog_id,
                'predictions': predictions
            }

# Report the final accuracy for this validation

# avg_val_accuracy_classification = total_eval_accuracy_classification / len(validation_dataloader)
# avg_val_accuracy_multilabel = total_eval_accuracy_multilabel / len(validation_dataloader)
avg_val_accuracy_classification = total_eval_accuracy_classification['matched'] / total_eval_accuracy_classification[
    'counts']
avg_val_accuracy_multilabel = total_eval_accuracy_multilabel['matched'] / total_eval_accuracy_multilabel['counts']
print("  Accuracy for classification (actions): {0:.4f}".format(avg_val_accuracy_classification))
print("  Accuracy for multilabel-classification (attributes): {0:.4f}".format(avg_val_accuracy_multilabel))

# Reference implementation: evaluation of action prediction along with attributes
metrics = evaluation.evaluate_action_prediction(devtest_dials, model_actions.values())
# print("model_actions passed to the evaluator:")
# for v in model_actions.values():
#   print(v)
print("***************************************")
print("Reference evaluation metrics:")
print(metrics)

# Storing results

# Display floats with two decimal places.
pd.set_option('precision', 2)
# Convert test data to dataframe
df_test = pd.DataFrame(data=test_batch)
# Create a DataFrame from our training statistics.
df_stats = pd.DataFrame(data=training_stats)
# Use the 'epoch' as the row index.
df_stats = df_stats.set_index('epoch')
# Save metrics on evaluation
metrics['params'] = exec_params
df_eval = pd.DataFrame(data=pd.json_normalize(metrics))

# Objects serialization
timestr = time.strftime("%Y%m%d-%H%M%S")
testdata_filename = f"./results/testdata-{timestr}"
stats_filename = f"./results/stats-{timestr}"
evaluation_filename = f"./results/eval-{timestr}"
df_test.to_pickle(testdata_filename)
df_stats.to_pickle(stats_filename)
df_eval.to_pickle(evaluation_filename)
