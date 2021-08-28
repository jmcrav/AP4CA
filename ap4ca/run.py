from absl import flags
from absl import app
from model import runner as r
from random import randint

# Set flags
FLAGS = flags.FLAGS

flags.DEFINE_integer(name="seed",
                     default=randint(100, 1000000),
                     help="Set for deterministic behaviour")
flags.DEFINE_string(name="results_path",
                    default="ap4ca/results",
                    help="Path for saving results",
                    short_name="s")
flags.DEFINE_string(name="pretrained_set",
                    default="bert-base-uncased",
                    help="Pretained bert model to be used")
flags.DEFINE_bool(name="use_next",
                  default=True,
                  help="Concatenate with previous transcript")
flags.DEFINE_integer(name="batch_size",
                     default=12,
                     help="Dimension of loaders batches")
flags.DEFINE_integer(name="epochs",
                     default=6,
                     help="Number of epochs for training loop")
flags.DEFINE_integer(name="hidden_output_dim",
                     default=768,
                     help="Number of node for the last hidden layer")
flags.DEFINE_float(name="learning_rate",
                   default=5e-5,
                   help="Learning rate")
flags.DEFINE_float(name="tolerance",
                   default=1e-8,
                   help="Tolerance for the optimezer function")
flags.DEFINE_string(name="action_activation",
                    default='softmax',
                    help="Activation function for actions classification")
flags.DEFINE_enum(name="device_type",
                  default="GPU",
                  enum_values=["GPU", "TPU"],
                  help="Device type for tensor computing (GPU or TPU)")
flags.DEFINE_string(name="validation_data",
                    default="ap4ca/simmc/data/fashion_dev_dials_api_calls.json",
                    help="Path to validation data")
flags.DEFINE_string(name="training_data",
                    default="ap4ca/simmc/data/fashion_train_dials_api_calls.json",
                    help="Path to training data")
flags.DEFINE_string(name="test_data",
                    default="ap4ca/simmc/data/fashion_devtest_dials_api_calls.json",
                    help="Path to evaluation data")
flags.DEFINE_float(name="w_SpecifyInfo",
                   default=1.0,
                   help="Cross Entropy Loss weight for 'SpecifyInfo' action class")
flags.DEFINE_float(name="w_None",
                   default=1.0,
                   help="Cross Entropy Loss weight for 'None' action class")
flags.DEFINE_float(name="w_SearchDatabase",
                   default=1.0,
                   help="Cross Entropy Loss weight for 'SearchDatabase' action class")
flags.DEFINE_float(name="w_AddToCart",
                   default=1.0,
                   help="Cross Entropy Loss weight for 'AddToCart' action class")
flags.DEFINE_float(name="w_SearchMemory",
                   default=1.0,
                   help="Cross Entropy Loss weight for 'SearchMemory' action class")
flags.DEFINE_integer(name="num_of_samples",
                     default=1,
                     help="Number of samples to be generated")

def main(argv):

    for i in range(1, FLAGS.num_of_samples):

        print("***************************************")
        print("Running BERT classifier")
        print(f"\n\tRun number {i}\n\n")

        # Build runner
        print("***************************************")
        print(f"\tBuild runner object")
        runner = r.Runner()

        # Train and eval model
        print("***************************************")
        print("\tCall training step")
        runner.train_and_eval()

        # Validate on test set
        print("***************************************")
        print("\tCall evaluation step")
        runner.validate()

        # Store results
        print("***************************************")
        print("\tStore results")
        runner.store_results()

if __name__ == '__main__':
    # Mark flag required for command line
    app.run(main)