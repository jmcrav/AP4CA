from .model import runner as r
from absl import flags
from absl import app
import flags as main_flags
from .model import flags as model_flags
from .simmc import flags as simmc_flags

# FIXME importare flags da __init__ o definendo in options???
main_flags.define()
model_flags.define()
simmc_flags.define()

FLAGS = flags.FLAGS

def main(argv):

    # Build runner
    runner = r.Runner()

    # Train and eval model
    runner.train_and_eval()

    # Validate on test set
    runner.validate()

    # Plot results
    runner.plot_result()

    # Store results
    runner.store_results()

if __name__ == '__main__':
    # Mark flag required for command line
    app.run(main)