from model import utils
from absl import flags
from absl import app
import flags as main_opts
from model import flags as model_opts
from model import utils
from simmc import flags as simmc_opt

# FIXME importare flags da __init__ o definendo in options???
main_opts.define()
model_opts.define()
simmc_opt.define()

FLAGS = flags.FLAGS

def main(argv):
    # Delete argv or it is used??
    utils.training_and_eval()

if __name__ == '__main__':
    # Mark flag required for command line
    app.run(main)