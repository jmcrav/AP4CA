from model import utils
from absl import flags
from absl import app
import flags as main_flags
from model import flags as model_flags
from model import utils as model_utils
from simmc import flags as simmc_flags

# FIXME importare flags da __init__ o definendo in options???
main_flags.define()
model_flags.define()
simmc_flags.define()

FLAGS = flags.FLAGS

def main(argv):

    # Delete argv or it is used??
    utils.training_and_eval()

if __name__ == '__main__':
    # Mark flag required for command line
    app.run(main)