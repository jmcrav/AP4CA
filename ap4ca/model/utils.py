import torch
from torch import nn
from absl import flags
import datetime

FLAGS = flags.FLAGS

def get_device():
    # Device from flag enum (GPU or TPU)
    if FLAGS.device_type=="GPU":
        # TODO check and raise error
        device = torch.device("cuda")
        print(f"There are {torch.cuda.device_count()} GPU(s) available.")
        print(f"We will use the GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        return device
    elif FLAGS.device_type=="TPU":
        # TODO TPU implementation
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
    loss_attributes_fn = nn.BCELoss()
    loss_actions = loss_actions_fn(actions_logits, actions_labels)
    loss_attributes = loss_attributes_fn(attributes_logits, attributes_labels.float())
    return loss_actions + loss_attributes

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
