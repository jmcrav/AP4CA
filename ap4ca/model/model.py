from transformers import BertModel
import torch
from torch import nn
from absl import flags

FLAGS = flags.FLAGS

class CustomBERTModel(nn.Module):

    def __init__(self,
                 actions_label_dim,
                 attributes_label_dim
                 ):
        """Model constructor

        :param actions_label_dim: numbers of action labels
        :param attributes_label_dim: numbers of attributes labels
        """
        super(CustomBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(FLAGS.pretrained_set)
        ### New layers:
        self.linear_intermedio = nn.Linear(768, FLAGS.hidden_output_dim)
        self.linear_actions = nn.Linear(FLAGS.hidden_output_dim, actions_label_dim)
        self.linear_attributes = nn.Linear(FLAGS.hidden_output_dim, attributes_label_dim)

    def forward(self, ids, mask):
        # controllare che l'output non rappresenti solo lo stato interno dovuto al token CLS
        output = self.bert(ids, attention_mask=mask)

        # prendiamo il campo last_hidden_state dall'oggetto output; last hidden state rappresenta il tensore
        # in uscita dallo step di forward del BertModel
        last_hidden_state_output = output["last_hidden_state"]
        # last_hidden_state has the following shape: (batch_size, sequence_length, 768)
        # stiamo passando solo il token CLS ai layer successivi
        linear_output_intermedio = self.linear_intermedio(last_hidden_state_output[:, 0, :].view(-1, 768))
        # linear_output_intermedio = self.linear_intermedio(pooled_output)

        linear_output_actions = self.linear_actions(linear_output_intermedio)
        # Test sigmoid for increasing perplexity performance
        if (FLAGS.action_activation=="softmax"):
            linear_output_actions = nn.functional.softmax(linear_output_actions, dim=1)
        else:
            linear_output_actions = torch.sigmoid(linear_output_actions)
        linear_output_attributes = self.linear_attributes(linear_output_intermedio)
        linear_output_attributes = torch.sigmoid(linear_output_attributes)

        return {'actions': linear_output_actions, 'attributes': linear_output_attributes}
