from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import torch
import numpy as np
from absl import flags
import random
import model.utils as md_utils
import model.loaders as loaders
from model.model import CustomBERTModel
from simmc import utils as simmc_utils
from simmc import action_evaluation as evaluation
import time
import pandas as pd

FLAGS = flags.FLAGS

class Runner():

    def __init__(self,
                 model=None,
                 optimizer=None,
                 scheduler=None
                 ):
        """Model runner constructor

        :param model: The custom BERT model to be run
        :param device: The device (GPU or TPU) for tensors manipulation
        """
        # Set random value for deterministic behaviour
        torch.manual_seed(FLAGS.seed)
        # Computing device (GPU or TPU)
        self.device = md_utils.get_device()
        # We'll store a number of quantities such as training and validation loss,
        # validation accuracy, and timings.
        self.training_stats = []
        self.test_batch = []
        # Parameters for training loop
        self.batch = FLAGS.batch_size
        self.epochs = FLAGS.epochs
        # Load data from json archives
        print("\tLoad input data")
        self.train_data = simmc_utils.get_data("train", format="df")
        self.validation_data = simmc_utils.get_data("val", format="df")
        self.test_data = simmc_utils.get_data("test", format="df")
        # BERT tokenizer
        print("\tLoad tokenizer")
        self.tokenizer = BertTokenizer.from_pretrained(FLAGS.pretrained_set, do_lower_case=True)
        # Labels encoders
        self.le = LabelEncoder()
        self.mlb = MultiLabelBinarizer()
        # Compute max length
        print("\tCompute transcripts max length")
        max_len_training = self.compute_max_len(self.train_data)
        print(f"\tMax length training data transcripts: {max_len_training}")
        max_len_validation = self.compute_max_len(self.validation_data)
        print(f"\tMax length validation data transcripts: {max_len_validation}")
        max_len_test = self.compute_max_len(self.test_data)
        print(f"\tMax length test data transcripts: {max_len_test}")
        self.max_len = max(max_len_training, max_len_validation, max_len_test)
        print(f"\tMax length for padding: {self.max_len}")
        # Encoding labels
        print("\tEncoding data and build dataloader")
        self.encoded_labels = self.label_encoding()
        print("\t\tBuild train dataloader")
        encoded_data = self.transcript_encoding(self.train_data)
        self.trainDataLoader=loaders.get_dataloader(
            input_ids=encoded_data['input_ids'],
            attention_mask=encoded_data['attention_mask'],
            labels_actions=self.encoded_labels['tr_act'],
            labels_attributes=self.encoded_labels['tr_att'],
            dialog_ids=torch.tensor(self.train_data.dialog_id.values, device=self.device),
            turn_idx=torch.tensor(self.train_data.turn_idx.values, device=self.device),
            batch_size=self.batch,
            type='random'
        )
        print("\t\tBuild validation dataloader")
        encoded_data = self.transcript_encoding(self.validation_data)
        self.validationDataLoader=loaders.get_dataloader(
            input_ids=encoded_data['input_ids'],
            attention_mask=encoded_data['attention_mask'],
            labels_actions=self.encoded_labels['vd_act'],
            labels_attributes=self.encoded_labels['vd_att'],
            dialog_ids=torch.tensor(self.validation_data.dialog_id.values, device=self.device),
            turn_idx=torch.tensor(self.validation_data.turn_idx.values, device=self.device),
            batch_size=self.batch,
            type='sequential'
        )
        print("\t\tBuild test dataloader")
        encoded_data = self.transcript_encoding(self.test_data)
        self.testDataLoader=loaders.get_dataloader(
            input_ids=encoded_data['input_ids'],
            attention_mask=encoded_data['attention_mask'],
            labels_actions=self.encoded_labels['tst_act'],
            labels_attributes=self.encoded_labels['tst_att'],
            dialog_ids=torch.tensor(self.test_data.dialog_id.values, device=self.device),
            turn_idx=torch.tensor(self.test_data.turn_idx.values, device=self.device),
            batch_size=self.batch,
            type = 'sequential'
        )

        self.total_step = len(self.trainDataLoader) + self.epochs

        print("\tBuild or load BERT model")
        if model==None:
            print(f"Actions classes: {len(self.le.classes_)} - Attributes classes: {len(self.mlb.classes_)}")
            self.model=CustomBERTModel(len(self.le.classes_), len(self.mlb.classes_))
            self.optimizer=AdamW(self.model.parameters(),
                                 lr=FLAGS.learning_rate,
                                 eps=FLAGS.tolerance)
            self.scheduler=get_linear_schedule_with_warmup(self.optimizer,
                                                           num_warmup_steps=0,
                                                           num_training_steps=len(self.trainDataLoader) * self.epochs)
        else:
            self.model=model
            self.optimizer=optimizer
            self.scheduler=scheduler

        if FLAGS.device_type == "GPU":
            self.model.cuda(self.device)

        self.metrics_eval = None

        # Run identifier
        self.timestr = time.strftime("%Y%m%d-%H%M%S")

        # Class action weights for cross entropy loss function (cost-sensitive learning for class imbalance problem)
        self.weights = torch.tensor(
            [FLAGS.w_AddToCart, FLAGS.w_None, FLAGS.w_SearchDatabase, FLAGS.w_SearchMemory, FLAGS.w_SpecifyInfo],
            device=self.device
        )

    def compute_max_len(self, df):
        """Compute tokenized transcripts max length

        :param df: pandas dataframe with the transcript to be tokenized
        :return: max length found in the dataset
        """

        max_len = 0
        for i in range(len(df)):
            if df.previous_transcript[i] != "" and FLAGS.use_next:
                input_ids = self.tokenizer.encode(
                    f"{df.previous_transcript[i]} {df.previous_system_transcript[i]}",
                    df.transcript[i],
                    add_special_tokens=True)
            else:
                input_ids = self.tokenizer.encode(df.transcript[i], add_special_tokens=True)
            max_len = max(max_len, len(input_ids))

        return max_len

    def transcript_encoding(self, df):
        """Tokenization of the dataframe sentences

        :param df: pandas dataframe to be encoded
        :param max_len: max length for sentences padding
        :return: a dictionary with two list, input_ids and attention_mask
        """
        # For every sentence `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.

        encoded_dict = [self.tokenizer.encode_plus(f"{df.previous_transcript[i]} {df.previous_system_transcript[i]}",
                                              # sentence to encode
                                              df.transcript[i],  # next sentence to encode
                                              add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                              truncation=True,
                                              max_length=self.max_len,  # Pad & truncate all sentences.
                                              padding='max_length',
                                              return_attention_mask=True,  # Build attention masks.
                                              return_tensors='pt'  # return pytorch tensors.
                                              )
                        if df.previous_transcript[i] != "" and FLAGS.use_next
                        else self.tokenizer.encode_plus(df.transcript[i],  # next sentence to encode
                                                   add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                                   truncation=True,
                                                   max_length=self.max_len,  # Pad & truncate all sentences.
                                                   padding='max_length',
                                                   return_attention_mask=True,  # Build attention masks.
                                                   return_tensors='pt'  # return pytorch tensors.
                                                   )
                        for i in range(len(df))]

        return {'input_ids': torch.cat([dict['input_ids'] for dict in encoded_dict], dim = 0),
                'attention_mask': torch.cat([dict['attention_mask'] for dict in encoded_dict], dim = 0)}

    def label_encoding(self):
        """Encoding actions and attributes labels

        :return: a dictionary with the encoded actions and attributes lists
        """
        attributes_labels = np.concatenate((self.train_data.attributes.values,
                                            self.validation_data.attributes.values,
                                            self.test_data.attributes.values),
                                           axis=None)
        attributes_yt = self.mlb.fit_transform(attributes_labels)
        return {
            'tr_act': torch.tensor(self.le.fit_transform(self.train_data.action.values), device=self.device),
            'tr_att': torch.tensor(attributes_yt[0:len(self.train_data.attributes.values)], device=self.device),
            'vd_act': torch.tensor(self.le.fit_transform(self.validation_data.action.values), device=self.device),
            'vd_att': torch.tensor(attributes_yt[len(self.train_data.attributes.values): len(self.train_data.attributes.values) + len(
                self.validation_data.attributes.values)], device=self.device),
            'tst_act': torch.tensor(self.le.fit_transform(self.test_data.action.values), device=self.device),
            'tst_att': torch.tensor(attributes_yt[len(self.train_data.attributes.values) + len(self.validation_data.attributes.values):],
                                    device=self.device)
        }

    def train_and_eval(self):
        """Training and evaluation steps

        :return:
        """
        dev_dials = simmc_utils.get_data("val", format="json")

        # Set the seed value all over the place to make this reproducible.
        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)
        # torch.manual_seed(FLAGS.seed)  must be done before RandomSampler instantiation
        if FLAGS.device_type == "GPU":
            torch.cuda.manual_seed_all(FLAGS.seed)

        # This training code is based on the `run_glue.py` script here:
        # https://github.com/huggingface/transformer    # Delete argv or it is used??s/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        # For each epoch...
        for epoch_i in range(0, self.epochs):
            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
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
            for step, batch in enumerate(self.trainDataLoader):

                # Progress update every 40 batches.
                if step % 400 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = md_utils.format_time(time.time() - t0)

                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(self.trainDataLoader), elapsed))

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

                loss = md_utils.loss_fn(result, b_labels_actions, b_labels_attributes, self.weights)

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
            avg_train_loss = total_train_loss / len(self.trainDataLoader)

            # Measure how long this epoch took.
            training_time = md_utils.format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.
            # TODO join validation and test

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.mlb.inverse_transform(attr_yt[3].reshape(1, -1))
            self.model.eval()

            # Tracking variables
            total_eval_accuracy_classification = {'matched': 0, 'counts': 0}
            total_eval_accuracy_multilabel = {'matched': 0, 'counts': 0}
            total_eval_loss = 0
            nb_eval_steps = 0

            batch_number = 0

            # Dictionary for action_evaluation
            model_actions = {}

            # Evaluate data for one epoch
            for batch in self.validationDataLoader:

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
                loss = md_utils.loss_fn(result, b_labels_actions, b_labels_attributes, self.weights)

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
                self.test_batch.append({
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
                        action_log_prob[self.le.classes_[act_i]] = np.log(actions_logits_foracc[el_i][act_i])
                    # attributes = {}
                    attributes = []
                    # attributes_list = np.rint(attributes_logits_foracc[el_i])
                    attributes_list = np.array(attributes_logits_foracc[el_i])
                    for attr in range(len(attributes_list)):
                        attribute = self.mlb.classes_[attr]
                        # attributes[mlb.classes_[attr]] = attributes_list[attr]
                        if attributes_list[attr] >= 0.5:
                            attributes.append(attribute)
                    prediction = {
                        'action': self.le.classes_[np.argmax(actions_logits_foracc[el_i])],
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
            avg_val_loss = total_eval_loss / len(self.validationDataLoader)

            # Measure how long the validation run took.
            validation_time = md_utils.format_time(time.time() - t0)

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

        print("Total training took {:} (h:mm:ss)".format(md_utils.format_time(time.time() - total_t0)))

    def validate(self):
        """
        TODO Check input file
        :param self:
        :return:
        """
        # with open('./extr_output/fashion_devtest_dials_api_calls.json') as f:
        #    devtest_dials = json.load(f)
        devtest_dials = simmc_utils.get_data("test", format="json")

        model_actions = {}

        # Tracking variables
        total_eval_accuracy_classification = {'matched': 0, 'counts': 0}
        total_eval_accuracy_multilabel = {'matched': 0, 'counts': 0}

        # Put model in evaluation mode
        self.model.eval()

        for batch in self.testDataLoader:

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
                    action_log_prob[self.le.classes_[act_i]] = np.log(actions_logits_foracc[el_i][act_i])
                # attributes = {}
                attributes = []
                # attributes_list = np.rint(attributes_logits_foracc[el_i])
                attributes_list = np.array(attributes_logits_foracc[el_i])
                for attr in range(len(attributes_list)):
                    attribute = self.mlb.classes_[attr]
                    # attributes[mlb.classes_[attr]] = attributes_list[attr]
                    if attributes_list[attr] >= 0.5:
                        attributes.append(attribute)
                prediction = {
                    'action': self.le.classes_[np.argmax(actions_logits_foracc[el_i])],
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
        avg_val_accuracy_classification = total_eval_accuracy_classification['matched'] / \
                                          total_eval_accuracy_classification['counts']
        avg_val_accuracy_multilabel = total_eval_accuracy_multilabel['matched'] / total_eval_accuracy_multilabel[
            'counts']
        print("  Accuracy for classification (actions): {0:.4f}".format(avg_val_accuracy_classification))
        print("  Accuracy for multilabel-classification (attributes): {0:.4f}".format(avg_val_accuracy_multilabel))

        # Reference implementation: evaluation of action prediction along with attributes
        self.metrics_eval = evaluation.evaluate_action_prediction(devtest_dials, model_actions.values())
        # print("model_actions passed to the evaluator:")
        # for v in model_actions.values():
        #   print(v)
        print("***************************************")
        print("Reference evaluation metrics:")
        print(self.metrics_eval)

    def store_results(self):
        category = f"B{FLAGS.batch_size}-E{FLAGS.epochs}-H{FLAGS.hidden_output_dim}-LR {FLAGS.learning_rate} - eps {FLAGS.tolerance} - ACT - {FLAGS.action_activation}"
        # Display floats with two decimal places.
        pd.set_option('precision', 2)
        # Convert test data to dataframe
        df_test = pd.DataFrame(data=self.test_batch)
        df_test['category'] = category
        # Create a DataFrame from our training statistics.
        df_stats = pd.DataFrame(data=self.training_stats)
        df_stats['category'] = category
        # Use the 'epoch' as the row index.
        df_stats = df_stats.set_index('epoch')
        # Save metrics on evaluation
        df_eval = pd.DataFrame(data=pd.json_normalize(self.metrics_eval))
        df_eval['category'] = category
        # Save parameters used
        df_params = pd.DataFrame(data=FLAGS.flag_values_dict())
        # Save classes
        classes_dict = {'actions' : self.le.classes_, 'attributes' : self.mlb.classes_}
        df_classes = pd.DataFrame(data=self.le.classes_)
        # Objects serialization
        testdata_filename = f"{FLAGS.results_path}/testdata-{self.timestr}"
        stats_filename = f"{FLAGS.results_path}/stats-{self.timestr}"
        evaluation_filename = f"{FLAGS.results_path}/eval-{self.timestr}"
        flags_filename = f"{FLAGS.results_path}/flags-{self.timestr}"
        classes_filename = f"{FLAGS.results_path}/classes-{self.timestr}"
        df_test.to_pickle(testdata_filename)
        df_stats.to_pickle(stats_filename)
        df_eval.to_pickle(evaluation_filename)
        df_params.to_pickle(flags_filename)
        df_classes.to_pickle(classes_filename)

