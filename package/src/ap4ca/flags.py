from absl import flags
from random import randint

def define():
    """Define command line arguments"""
    try:
        flags.DEFINE_integer(name="seed",
                             default=randint(100, 1000000),
                             help="Set for deterministic behaviour")
    except flags.DuplicateFlagError:
        pass