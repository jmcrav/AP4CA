from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn
from absl import flags

FLAGS = flags.FLAGS

class CustomBERTModel(nn.Module):

    def __init__(self,
                 actions_label_dim,
                 attributes_label_dim,
                 hidden_output_dim=256,
                 pretrained_set="bert-base-uncased"
                 ):
        """Model constructor

        :param actions_label_dim: numbers of action labels
        :param attributes_label_dim: numbers of attributes labels
        :param hidden_output_dim: dimension of the hidden layers added to the base BERT model (default 256)
        :param pretrained_set: BERT pretained set (default bert-base-uncased)
        """
        super(CustomBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_set)
        ### New layers:
        self.linear_intermedio = nn.Linear(768, hidden_output_dim)
        # TODO test adding other layers with different dimensions and/or type (non linear?)
        self.linear_actions = nn.Linear(hidden_output_dim, actions_label_dim)
        self.linear_attributes = nn.Linear(hidden_output_dim, attributes_label_dim)

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
        # linear_output_actions = self.sftmx(linear_output_actions)
        # linear_output_actions = nn.functional.softmax(linear_output_actions)
        # Test sigmoid for increasing perplexity performance
        # linear_output_actions = torch.sigmoid(linear_output_actions)
        linear_output_actions = nn.functional.relu(linear_output_actions)
        linear_output_actions = nn.functional.softmax(linear_output_actions, dim=1)
        linear_output_attributes = self.linear_attributes(linear_output_intermedio)
        # linear_output_attributes = self.sig(linear_output_attributes)
        linear_output_attributes = torch.sigmoid(linear_output_attributes)

        return {'actions': linear_output_actions, 'attributes': linear_output_attributes}
