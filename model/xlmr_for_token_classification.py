from fairseq.models.roberta import XLMRModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import sys
sys.path.append("/home/dunghc/xlm-roberta-ner")
from neural_types import LabelsType, LogitsType, LogprobsType, LossType, MaskType, NeuralType
from classes.common import Serialization, Typing, typecheck
from classes.loss import Loss

class CrossEntropyLoss(nn.CrossEntropyLoss, Serialization, Typing):
    """
    CrossEntropyLoss
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "logits": NeuralType(['B'] + ['ANY'] * (self._logits_dim - 1), LogitsType()),
            "labels": NeuralType(['B'] + ['ANY'] * (self._logits_dim - 2), LabelsType()),
            "loss_mask": NeuralType(['B'] + ['ANY'] * (self._logits_dim - 2), MaskType(), optional=True),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, logits_ndim=2, weight=None, reduction='mean', ignore_index=-100):
        """
        Args:
            logits_ndim (int): number of dimensions (or rank) of the logits tensor
            weight (list): list of rescaling weight given to each class
            reduction (str): type of the reduction over the batch
        """
        if weight is not None and not torch.is_tensor(weight):
            weight = torch.FloatTensor(weight)
            logging.info(f"Weighted Cross Entropy loss with weight {weight}")
        super().__init__(weight=weight, reduction=reduction, ignore_index=ignore_index)
        self._logits_dim = logits_ndim

    @typecheck()
    def forward(self, logits, labels, loss_mask=None):
        """
        Args:
            logits (float): output of the classifier
            labels (long): ground truth labels
            loss_mask (bool/float/int): tensor to specify the masking
        """
        logits_flatten = torch.flatten(logits, start_dim=0, end_dim=-2)
        labels_flatten = torch.flatten(labels, start_dim=0, end_dim=-1)

        if loss_mask is not None:
            if loss_mask.dtype is not torch.bool:
                loss_mask = loss_mask > 0.5
            loss_mask_flatten = torch.flatten(loss_mask, start_dim=0, end_dim=-1)
            logits_flatten = logits_flatten[loss_mask_flatten]
            labels_flatten = labels_flatten[loss_mask_flatten]

        if len(labels_flatten) == 0:
            return super().forward(logits, torch.argmax(logits, dim=-1))

        loss = super().forward(logits_flatten, labels_flatten)
        return loss

class AggregatorLoss(Loss):
    """
    Sums several losses into one.

    Args:
        num_inputs: number of input losses
        weights: a list of coefficient for merging losses
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        input_types = {}
        for i in range(self._num_losses):
            input_types["loss_" + str(i + 1)] = NeuralType(elements_type=LossType())

        return input_types

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, num_inputs: int = 2, weights: List[float] = None):
        super().__init__()
        self._num_losses = num_inputs
        if weights is not None and len(weights) != num_inputs:
            raise ValueError("Length of weights should be equal to the number of inputs (num_inputs)")

        self._weights = weights

    @typecheck()
    def forward(self, **kwargs):
        values = [kwargs[x] for x in sorted(kwargs.keys())]
        loss = torch.zeros_like(values[0])
        for loss_idx, loss_value in enumerate(values):
            if self._weights is not None:
                loss = loss.add(loss_value, alpha=self._weights[loss_idx])
            else:
                loss = loss.add(loss_value)
        return loss

class XLMRForTokenClassification(nn.Module):

    def __init__(self, pretrained_path, n_capit_labels, n_punc_labels, hidden_size, dropout_p, label_ignore_idx=0,
                head_init_range=0.04, device='cuda'):
        super().__init__()

        self.n_capit_labels = n_capit_labels
        self.n_punc_labels = n_punc_labels
        
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        # self.classification_head = nn.Linear(hidden_size, n_labels)
        self.classification_punc_head = nn.Linear(hidden_size, n_punc_labels)
        self.classification_capit_head = nn.Linear(hidden_size, n_capit_labels)
        
        self.label_ignore_idx = label_ignore_idx

        self.xlmr = XLMRModel.from_pretrained(pretrained_path)
        self.model = self.xlmr.model
        self.dropout = nn.Dropout(dropout_p)

        # loss
        self.loss = CrossEntropyLoss(logits_ndim=2, ignore_index=self.label_ignore_idx)
        self.agg_loss = AggregatorLoss(num_inputs=2)

        
        self.device=device

        # initializing classification head
        # self.classification_head.weight.data.normal_(mean=0.0, std=head_init_range)
        self.classification_punc_head.weight.data.normal_(mean=0.0, std=head_init_range)
        self.classification_capit_head.weight.data.normal_(mean=0.0, std=head_init_range)

    def forward(self, inputs_ids, labels_capit, labels_punc, labels_mask, valid_mask):
        '''
        Computes a forward pass through the sequence tagging model.
        Args:
            inputs_ids: tensor of size (bsz, max_seq_len). padding idx = 1
            labels: tensor of size (bsz, max_seq_len)
            labels_mask and valid_mask: indicate where loss gradients should be propagated and where 
            labels should be ignored

        Returns :
            logits: unnormalized model outputs.
            loss: Cross Entropy loss between labels and logits

        '''
        transformer_out, _ = self.model(inputs_ids, features_only=True)

        out_1 = F.relu(self.linear_1(transformer_out))
        # print("out_1: ", out_1.shape)
        out_1 = self.dropout(out_1)
        # print("out_1: ", out_1.shape)
        logits_capit = self.classification_capit_head(out_1)
        logits_punc = self.classification_punc_head(out_1)

        if labels_capit is not None and labels_punc is not None:
            # loss_fct = nn.CrossEntropyLoss(ignore_index=self.label_ignore_idx)
            # Only keep active parts of the loss
            if labels_mask is not None:
                # capit loss
                active_loss = valid_mask.view(-1) == 1
                active_logits = logits_capit.view(-1, self.n_capit_labels)[active_loss]
                active_labels = labels_capit.view(-1)[active_loss]
                loss_capit = self.loss(logits=active_logits, labels=active_labels)

                # punc loss
                active_loss = valid_mask.view(-1) == 1
                active_logits = logits_punc.view(-1, self.n_punc_labels)[active_loss]
                active_labels = labels_punc.view(-1)[active_loss]
                loss_punc = self.loss(logits=active_logits, labels=active_labels)

                # sum loss
                loss = self.agg_loss(loss_1=loss_capit, loss_2=loss_punc)

                # punct_loss  = self.loss()
                #print("Preds = ", active_logits.argmax(dim=-1))
                #print("Labels = ", active_labels)
            else:
                # loss = loss_fct(
                #     logits.view(-1, self.n_labels), labels.view(-1))
                loss1 = self.loss(logits=active_logits, labels=active_labels)
                loss2 = self.loss(logits=active_logits, labels=active_labels)
                loss = self.agg_loss(loss_1=loss1, loss_2=loss2)

            return loss
        else:
            return logits_capit, logits_punc

    def encode_word(self, s):
        """
        takes a string and returns a list of token ids
        """
        tensor_ids = self.xlmr.encode(s)
        # remove <s> and </s> ids
        return tensor_ids.cpu().numpy().tolist()[1:-1]
