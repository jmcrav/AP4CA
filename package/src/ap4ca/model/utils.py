from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from absl import flags
import datetime

FLAGS = flags.FLAGS

def get_device():
    if FLAGS.device=="GPU":
        pass
    elif FLAGS.device=="TPU":
        pass

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
                   device=get_device()):
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

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
