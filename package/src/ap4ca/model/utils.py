from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn
import numpy as np
from sklearn.prepocessing import MultiLabelBinarizer, LabelEncoder
from absl import flags
import datetime

FLAGS = flags.FLAGS


def loss_fn(logits,
            actions_labels,
            attributes_labels):
    """Loss function definition

    :param logits:
    :param actions_labels:
    :param attributes_labels:
    :return:
    """
    actions_logits = logits['actions']
    attributes_logits = logits['attributes']
    loss_actions_fn = nn.CrossEntropyLoss()
    # loss_attributes_fn = nn.BCELoss()
    loss_attributes_fn = nn.BCEWithLogitsLoss()
    loss_actions = loss_actions_fn(actions_logits, actions_labels)
    loss_attributes = loss_attributes_fn(attributes_logits, attributes_labels.float())
    return loss_actions + loss_attributes

def label_encoding(df_tr,
                   df_vd,
                   df_tst,
                   device):
    """Encoding actions and attributes labels

    :param df_tr: pandas training dataframe
    :param df_vd: pandas validation dataframe
    :param df_tst: pandas test dataframe
    :return: a dictionary with the encoded actions and attributes lists
    """

    le = LabelEncoder()
    mlb = MultiLabelBinarizer()
    attributes_labels = np.concatenate((df_tr.attributes.values, df_vd.attributes.values, df_tst.attributes.values), axis=None)
    attributes_yt = mlb.fit_transform(attributes_labels)
    return {
        'tr_act': torch.tensor(le.fit_transform(df_tr.action.values), device=device),
        'tr_att' : torch.tensor(attributes_yt[0:len(df_tr.attributes.values)], device=device),
        'vd_act': torch.tensor(le.fit_transform(df_vd.action.values), device=device),
        'vd_att': torch.tensor(attributes_yt[len(df_tr.attributes.values), len(df_tr.attributes.values) + len(df_vd.attributes.values)], device=device),
        'tst_act': torch.tensor(le.fit_transform(df_tst.action.values), device=device),
        'tst_att': torch.tensor(attributes_yt[len(df_tr.attributes.values) + len(df_vd.attributes.values):], device=device)
    }

def max_len(df,
            use_next=True,
            pretrained_set="bert-base-uncased"
            ):
    """Compute tokenized transcripts max length

    :param df: pandas dataframe with the transcript to be tokenized
    :param use_next: True for building utterances with transcripts from previous turns
    :return: max length found in the dataset
    """

    tokenizer = BertTokenizer.from_pretained(pretrained_set, do_lower_case=True)
    input_ids = [tokenizer.encode(f"{df.previous_transcript[i]} {df.previous_system_transcript[i]}", df.transcript[i],
                                  add_special_tokens=True)
                 if df.previous_transcript != "" and use_next
                 else tokenizer.encode(df.transcript[i], add_special_tokens=True)
                 for i in range(len(df))]

    return max(input_ids)

def df_encoding(df,
                max_len,
                use_next=True,
                pretrained_set="bert-base-uncased"
                ):
    """Tokenization of the dataframe sentences

    :param df: pandas dataframe to be encoded
    :param max_len: max length for sentences padding
    :param use_next: True for building utterances with transcripts from previous turns
    :return: a dictionary with two list, input_ids and attention_mask
    """
    # For every sentence `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.

    tokenizer = BertTokenizer.from_pretained(pretrained_set, do_lower_case=True)

    encoded_dict = [tokenizer.encode_plus(f"{df.previous_transcript[i]} {df.previous_system_transcript[i]}", # sentence to encode
                                          df.transcript[i], # next sentence to encode
                                          add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                                          truncation=True,
                                          max_length=max_len, # Pad & truncate all sentences.
                                          pad_to_max_length=True,
                                          return_attention_mask=True, # Build attention masks.
                                          return_tensors='pt' # return pytorch tensors.
                                          )
                    if df.previous_transcript[i] != "" and use_next
                    else tokenizer.encode_plus(df.transcript[i],  # next sentence to encode
                                               add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                               truncation=True,
                                               max_length=max_len,  # Pad & truncate all sentences.
                                               pad_to_max_length=True,
                                               return_attention_mask=True,  # Build attention masks.
                                               return_tensors='pt'  # return pytorch tensors.
                                               )
                    for i in range(len(df))]

    return {'input_ids': torch.cat(encoded_dict['input_ids'], dim=0),
            'attention_mask': torch.cat(encoded_dict['attention_mask'], dim=0)}


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
