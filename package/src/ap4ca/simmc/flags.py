from absl import flags

def define():
    """Define command line arguments"""
    flags.DEFINE_string("validation_data",
                        "./simmc/data/fashion_dev_dials_api_calls.json",
                        "Path to validation data")
    flags.DEFINE_string("training_data",
                        "./simmc/data/fashion_train_dials_api_calls.json",
                        "Path to training data")
    flags.DEFINE_string("test_data",
                        "./simmc/data/fashion_devtest_dials_api_calls.json",
                        "Path to evaluation data")
