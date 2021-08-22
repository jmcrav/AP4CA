from absl import flags
from random import randint

def define():
    """Define command line arguments"""
    flags.DEFINE_string(name="results_path",
                        default="./results",
                        help="Path for saving results",
                        short_name="s")
    try:
        flags.DEFINE_integer(name="seed",
                             default=randint(100, 1000000),
                             help="Set for deterministic behaviour")
    except flags.DuplicateFlagError:
        pass
    try:
        flags.DEFINE_integer(name="epochs",
                             default=4,
                             help="Number of epochs for training loop")
    except flags.DuplicateFlagError:
        pass
