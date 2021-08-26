"""Utilities for simmc data manipulation
"""

import pandas as pd
import json
import numpy as np
from absl import flags

FLAGS = flags.FLAGS

def createDataframe(json_file):
    """Dataframes builder

    :param json_file: path to file to be parsed
    :return: pandas dataframe
    """
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

    # Dataframe with turn id and various transcript useful to build composed utterances
    df = pd.DataFrame(data, columns=['dialog_id', 'turn_idx', 'transcript', 'action', 'attributes', 'system_transcript',
                                     'transcript_annotated', 'system_transcript_annotated', 'previous_transcript',
                                     'previous_system_transcript'])

    df.sort_values(by=['dialog_id', 'turn_idx'])
    df.loc[1:, 'previous_transcript'] = [df.loc[i - 1, 'transcript']
                                         if df['dialog_id'][i] == df['dialog_id'][i - 1]
                                         else ''
                                         for i in range(1, len(df))]
    df.loc[1:, 'previous_system_transcript'] = [df.loc[i - 1, 'system_transcript']
                                                if df['dialog_id'][i] == df['dialog_id'][i - 1]
                                                else ''
                                                for i in range(1, len(df))]

    return df

def flat_accuracy_actions(preds, labels):
    """Function to calculate the accuracy of our actions predictions vs actual labels

    :param preds: actions predictions tensor
    :param labels: labels tensor
    :return: a dictionary with number of matched actions predictions and total count
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return {'matched': np.sum(pred_flat == labels_flat), 'counts': len(labels_flat)}

def flat_accuracy_attributes(preds, labels):
    """Function to calculate the accuracy of our attributes predictions vs actual labels

    :param preds: attributes predictions tensor
    :param labels: labels tensor
    :return: a dictionary with number of matched attribute predictions and total count
    """
    tot_preds = preds.shape[0]
    preds_int = np.rint(preds)
    tot_eq = 0
    for i in range(tot_preds):
        comparison = preds_int[i] == labels[i]
        if comparison.all():
            tot_eq += 1
    return {'matched': tot_eq, 'counts' : tot_preds}

def get_data(type="train"):
    """Get data from simmc library

    :param type: train=training data, eval=evaluation data, val=validation data
    :return: json data parsed
    """
    if type=="train":
        return createDataframe(FLAGS.training_data)
    elif type=="val":
        return createDataframe(FLAGS.validation_data)
    elif type=="test":
        return createDataframe(FLAGS.test_data)
    else:
        raise ValueError("Data type must be: 'train' for training, 'val' for validation or 'test' for test data")
