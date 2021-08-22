from absl import flags

def define():
    """Define command line arguments"""
    flags.DEFINE_string("val_data",
                        "./simmc/data/fashion_dev_dials_api_calls.json",
                        "Path to validation data")
    flags.DEFINE_string("train_data",
                        "./simmc/data/fashion_train_dials_api_calls.json",
                        "Path to training data")
    flags.DEFINE_string("eval_data",
                        "./simmc/data/fashion_train_dials_api_calls.json",
                        "Path to evaluation data")
