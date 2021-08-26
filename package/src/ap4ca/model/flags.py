from absl import flags
from random import randint

def define():
    """Define command line arguments"""
    flags.DEFINE_string(name="results_path",
                        default="./results",
                        help="Path for saving results",
                        short_name="s")
    flags.DEFINE_string(name="pretrained_set",
                        default="bert-base-uncased",
                        help="Pretained bert model to be used")
    flags.DEFINE_bool(name="use_next",
                      default=True,
                      help="Concatenate with previous transcript")
    try:
        flags.DEFINE_integer(name="seed",
                             default=randint(100, 1000000),
                             help="Set for deterministic behaviour")
    except flags.DuplicateFlagError:
        pass
    try:
        flags.DEFINE_integer(name="batch_size",
                             default=32,
                             help="Dimension of loaders batches")
    except flags.DuplicateFlagError:
        pass
    try:
        flags.DEFINE_integer(name="epochs",
                             default=4,
                             help="Number of epochs for training loop")
    except flags.DuplicateFlagError:
        pass
    try:
        flags.DEFINE_integer(name="hidden_output_dim",
                             default=768,
                             help="Number of node for the last hidden layer")
    except flags.DuplicateFlagError:
        pass
    try:
        flags.DEFINE_float(name="learning_rate",
                           default=5e-5,
                           help="Learning rate")
    except flags.DuplicateFlagError:
        pass
    try:
        flags.DEFINE_float(name="tolerance",
                           default=1e-8,
                           help="Tolerance for the optimezer function")
    except flags.DuplicateFlagError:
        pass
    try:
        flags.DEFINE_string(name="action_activation",
                            default='softmax',
                            help="Activation function for actions classification")
    except flags.DuplicateFlagError:
        pass
    flags.DEFINE_enum(name="device_type",
                      default="GPU",
                      enum_values=['GPU', 'TPU'],
                      help="Device type for tensor computing (GPU or TPU)")
