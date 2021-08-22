from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn
import numpy as np
from absl import flags
import random
from . import utils as model_utils
from . import loaders
from ..simmc import utils as simmc_utils
from ..simmc import action_evaluation as evaluation
import time
import pandas as pd

FLAGS = flags.FLAGS

class Runner():

    def __init__(self,
                 model,
                 device
                 ):
        """Model runner constructor

        :param model: The custom BERT model to be run
        :param device: The device (GPU or TPU) for tensors manipulation
        """
        self.model = model
        # FIXME get type from FLAGS.device_type and build
        self.device = device
        # We'll store a number of quantities such as training and validation loss,
        # validation accuracy, and timings.
        self.training_stats = []
        self.test_batch = []
        self.model_actions = {}
        self.optimizer = None
        self.scheduler = None

    def train_and_eval(self):
        """Training and evaluation steps

        :return:
        """
        dev_dials = simmc_utils.get_data("train")

        train_dataloader = loaders.get_dataloader()

        # Set the seed value all over the place to make this reproducible.
        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)
        # torch.manual_seed(FLAGS.seed)  must be done before RandomSampler instantiation
        torch.cuda.manual_seed_all(FLAGS.seed)

        epochs = FLAGS.epochs

        # This training code is based on the `run_glue.py` script here:
        # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        # For each epoch...
        for epoch_i in range(0, epochs):
            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_loss = 0

            # Put the model into training mode. Don't be mislead--the call to
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            self.model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                # Progress update every 40 batches.
                if step % 400 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = model_utils.format_time(time.time() - t0)

                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: actions labels
                #   [3]: attributes labels
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels_actions = batch[2].to(self.device)
                b_labels_attributes = batch[3].to(self.device)

                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because
                # accumulating the gradients is "convenient while training RNNs".
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                self.model.zero_grad()

                # Perform a forward pass (evaluate the model on this training batch).
                # In PyTorch, calling `model` will in turn call the model's `forward`
                # function and pass down the arguments. The `forward` function is
                # documented here:
                # https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
                # The results are returned in a results object, documented here:
                # https://huggingface.co/transformers/main_classes/output.html#transformers.modeling_outputs.SequenceClassifierOutput
                # Specifically, we'll get the loss (because we provided labels) and the
                # "logits"--the model outputs prior to activation.
                result = self.model(b_input_ids,
                                    mask=b_input_mask)

                loss = model_utils.loss_fn(result, b_labels_actions, b_labels_attributes)

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value
                # from the tensor.
                total_train_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.from transformers import BertModel, BertConfig
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                self.optimizer.step()

                # Update the learning rate.
                self.scheduler.step()

            print(f"End of epoch {epoch_i}")

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)

            # Measure how long this epoch took.
            training_time = model_utils.format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.mlb.inverse_transform(attr_yt[3].reshape(1, -1))
            # FIXME reset model to train again
            self.model.eval()

            # Tracking variables
            total_eval_accuracy_classification = {'matched': 0, 'counts': 0}
            total_eval_accuracy_multilabel = {'matched': 0, 'counts': 0}
            total_eval_loss = 0
            nb_eval_steps = 0

            batch_number = 0

            # Dictionary for action_evaluation
            model_actions = {}

            validation_dataloader = loaders.get_dataloader()
            # Evaluate data for one epoch
            for batch in validation_dataloader:

                batch_number += 1

                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using
                # the `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels_actions = batch[2].to(self.device)
                b_labels_attributes = batch[3].to(self.device)
                b_dialog_ids = batch[4].to(self.device).detach().cpu().numpy()
                b_turn_idxs = batch[5].to(self.device).detach().cpu().numpy()

                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():

                    # Forward pass, calculate logit predictions.
                    # token_type_ids is the same as the "segment ids", which
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    result = self.model(b_input_ids,
                                   mask=b_input_mask)

                # Get the loss and "logits" output by the model. The "logits" are the
                # output values prior to applying an activation function like the
                # softmax.
                loss = model_utils.loss_fn(result, b_labels_actions, b_labels_attributes)

                # Accumulate the validation loss.
                total_eval_loss += loss.item()

                # Move logits and labels to CPU
                # logits = logits.detach().cpu().numpy()
                # label_ids = b_labels.to('cpu').numpy()

                actions_logits_foracc = result['actions'].detach().cpu().numpy()
                attributes_logits_foracc = result['attributes'].detach().cpu().numpy()
                actions_labels_foracc = b_labels_actions.to('cpu').numpy()
                attributes_labels_foracc = b_labels_attributes.to('cpu').numpy()

                # TODO: definire la nostra funzione di accuracy

                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                accuracy_classification = simmc_utils.flat_accuracy_actions(actions_logits_foracc, actions_labels_foracc)
                accuracy_multilabel = simmc_utils.flat_accuracy_attributes(attributes_logits_foracc, attributes_labels_foracc)

                total_eval_accuracy_classification['matched'] += accuracy_classification['matched']
                total_eval_accuracy_classification['counts'] += accuracy_classification['counts']
                total_eval_accuracy_multilabel['matched'] += accuracy_multilabel['matched']
                total_eval_accuracy_multilabel['counts'] += accuracy_multilabel['counts']
                # Salvo dati elaborazione batch per debug/analisi
                test_batch.append({
                    'epoch': epoch_i + 1,
                    'batchnum': batch_number,
                    'actions_logits': actions_logits_foracc,
                    'actions_labels': actions_labels_foracc,
                    'attributes_logits': attributes_logits_foracc,
                    'attributes_labels': attributes_labels_foracc,
                    'accuracy_classification': accuracy_classification,
                    'accuracy_multilabel': accuracy_multilabel
                })

                # Fill dictionary for action_evaluation
                for el_i in range(len(actions_logits_foracc)):
                    dialog_id = b_dialog_ids[el_i]
                    action_log_prob = {}
                    for act_i in range(len(actions_logits_foracc[el_i])):
                        # todo: controllare che la probabilità predetta sia in scala logaritmica (?? potrebbe essere fonte di errori)
                        action_log_prob[le.classes_[act_i]] = np.log(actions_logits_foracc[el_i][act_i])
                    # attributes = {}
                    attributes = []
                    # attributes_list = np.rint(attributes_logits_foracc[el_i])
                    attributes_list = np.array(attributes_logits_foracc[el_i])
                    for attr in range(len(attributes_list)):
                        attribute = mlb.classes_[attr]
                        # attributes[mlb.classes_[attr]] = attributes_list[attr]
                        if attributes_list[attr] >= 0.5:
                            attributes.append(attribute)
                    prediction = {
                        'action': le.classes_[np.argmax(actions_logits_foracc[el_i])],
                        'action_log_prob': action_log_prob,
                        'attributes': {'attributes': attributes},
                        'turn_id': b_turn_idxs[el_i]
                    }
                    if dialog_id in model_actions:
                        model_actions[dialog_id]['predictions'].append(prediction)
                    else:
                        predictions = list()
                        predictions.append(prediction)
                        model_actions[dialog_id] = {
                            'dialog_id': dialog_id,
                            'predictions': predictions
                        }

            # Report the final accuracy for this validation

            # avg_val_accuracy_classification = total_eval_accuracy_classification / len(validation_dataloader)
            # avg_val_accuracy_multilabel = total_eval_accuracy_multilabel / len(validation_dataloader)
            avg_val_accuracy_classification = total_eval_accuracy_classification['matched'] / \
                                              total_eval_accuracy_classification['counts']
            avg_val_accuracy_multilabel = total_eval_accuracy_multilabel['matched'] / total_eval_accuracy_multilabel['counts']
            print("  Accuracy for classification (actions): {0:.4f}".format(avg_val_accuracy_classification))
            print("  Accuracy for multilabel-classification (attributes): {0:.4f}".format(avg_val_accuracy_multilabel))

            # Reference implementation: evaluation of action prediction along with attributes
            metrics = evaluation.evaluate_action_prediction(dev_dials, model_actions.values())
            # print("model_actions passed to the evaluator:")
            print("***************************************")
            print("Reference evaluation metrics:")
            print(metrics)

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(validation_dataloader)

            # Measure how long the validation run took.
            validation_time = format_time(time.time() - t0)

            print("  Validation Loss: {0:.4f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            self.training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur. class.': avg_val_accuracy_classification,
                    'Valid. Accur. mult.label': avg_val_accuracy_multilabel,
                    'Training Time': training_time,
                    'Validation Time': validation_time,
                    'metrics': metrics
                }
            )

        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(model_utils.format_time(time.time() - total_t0)))

        def validate(self, act_classes, attr_classes):
            """
            TODO Check input file
            :param self:
            :return:
            """
            # with open('./extr_output/fashion_devtest_dials_api_calls.json') as f:
            #    devtest_dials = json.load(f)
            devtest_dials = simmc_utils.get_data("eval")

            evaluation_dataloader = loaders.get_dataloader()

            # Tracking variables
            total_eval_accuracy_classification = {'matched': 0, 'counts': 0}
            total_eval_accuracy_multilabel = {'matched': 0, 'counts': 0}

            # Put model in evaluation mode
            self.model.eval()

            for batch in evaluation_dataloader:

                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using
                # the `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels_actions = batch[2].to(device)
                b_labels_attributes = batch[3].to(device)
                b_dialog_ids = batch[4].to(device).detach().cpu().numpy()
                b_turn_idxs = batch[5].to(device).detach().cpu().numpy()

                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():
                    # Forward pass, calculate logit predictions.
                    # token_type_ids is the same as the "segment ids", which
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    result = self.model(b_input_ids, mask=b_input_mask)

                actions_logits_foracc = result['actions'].detach().cpu().numpy()
                attributes_logits_foracc = result['attributes'].detach().cpu().numpy()
                actions_labels_foracc = b_labels_actions.to('cpu').numpy()
                attributes_labels_foracc = b_labels_attributes.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                accuracy_classification = simmc_utils.flat_accuracy_actions(actions_logits_foracc,
                                                                            actions_labels_foracc)
                accuracy_multilabel = simmc_utils.flat_accuracy_attributes(attributes_logits_foracc,
                                                                           attributes_labels_foracc)

                total_eval_accuracy_classification['matched'] += accuracy_classification['matched']
                total_eval_accuracy_classification['counts'] += accuracy_classification['counts']
                total_eval_accuracy_multilabel['matched'] += accuracy_multilabel['matched']
                total_eval_accuracy_multilabel['counts'] += accuracy_multilabel['counts']

                # Fill dictionary for action_evaluation
                for el_i in range(len(actions_logits_foracc)):
                    dialog_id = b_dialog_ids[el_i]
                    action_log_prob = {}
                    for act_i in range(len(actions_logits_foracc[el_i])):
                        # todo: controllare che la probabilità predetta sia in scala logaritmica (?? potrebbe essere fonte di errori)
                        action_log_prob[le.classes_[act_i]] = np.log(actions_logits_foracc[el_i][act_i])
                    # attributes = {}
                    attributes = []
                    # attributes_list = np.rint(attributes_logits_foracc[el_i])
                    attributes_list = np.array(attributes_logits_foracc[el_i])
                    for attr in range(len(attributes_list)):
                        attribute = attr_classes[attr]
                        # attributes[mlb.classes_[attr]] = attributes_list[attr]
                        if attributes_list[attr] >= 0.5:
                            attributes.append(attribute)
                    prediction = {
                        'action': act_classes[np.argmax(actions_logits_foracc[el_i])],
                        'action_log_prob': action_log_prob,
                        'attributes': {'attributes': attributes},
                        'turn_id': b_turn_idxs[el_i]
                    }
                    if dialog_id in self.model_actions:
                        self.model_actions[dialog_id]['predictions'].append(prediction)
                    else:
                        predictions = list()
                        predictions.append(prediction)
                        self.model_actions[dialog_id] = {
                            'dialog_id': dialog_id,
                            'predictions': predictions
                        }

            # Report the final accuracy for this validation

            # avg_val_accuracy_classification = total_eval_accuracy_classification / len(validation_dataloader)
            # avg_val_accuracy_multilabel = total_eval_accuracy_multilabel / len(validation_dataloader)
            avg_val_accuracy_classification = total_eval_accuracy_classification['matched'] / \
                                              total_eval_accuracy_classification['counts']
            avg_val_accuracy_multilabel = total_eval_accuracy_multilabel['matched'] / total_eval_accuracy_multilabel[
                'counts']
            print("  Accuracy for classification (actions): {0:.4f}".format(avg_val_accuracy_classification))
            print("  Accuracy for multilabel-classification (attributes): {0:.4f}".format(avg_val_accuracy_multilabel))

            # Reference implementation: evaluation of action prediction along with attributes
            metrics = evaluation.evaluate_action_prediction(devtest_dials, model_actions.values())
            # print("model_actions passed to the evaluator:")
            # for v in model_actions.values():
            #   print(v)
            print("***************************************")
            print("Reference evaluation metrics:")
            print(metrics)

        def store_results(self):
            pass






