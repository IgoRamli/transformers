import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from ...activations import ACT2FN, get_activation
from ...file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import (
    BaseModelOutputWithCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import (
    PreTrainedModel,
    SequenceSummary,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from ...utils import logging
from .configuration_kepler import KeplerConfig


logger = logging.get_logger(__name__)

KEPLER_PRETRAINED_MODEL_ARCHIVE_LIST = []


class KeplerForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.KeplerForPreTraining`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss of the KEPLER objective (Discriminator + KE task).
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Prediction scores of the discriminator head (scores for each token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the discriminator embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    pScore: torch.FloatTensor = None
    nScore: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class KeplerKnowledgeEmbeddingPredictions(nn.Module):
    def __init__(self, config, encoder):
        super().__init__()

        self.encoder = encoder
        self.config = config
        self.nrelation = config.nrelation
        self.gamma = nn.Parameter(
            torch.Tensor([config.gamma]),
            requires_grad = False
        )
        self.eps = 2.0
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.eps) / config.embedding_size]),
            requires_grad = False
        )
        self.relation_embedding = nn.Embedding(config.nrelation, config.embedding_size)
        nn.init.uniform_(
            tensor = self.relation_embedding.weight,
            a = -self.embedding_range.item(),
            b = self.embedding_range.item()
        )

        model_func = {
            'TransE': self.TransE,
        }
        self.score_function = model_func[config.ke_model]

    def TransE(self, head, relation, tail):
        score = (head + relation) - tail
        score = self.gamma.item() - torch.norm(score, p=2, dim=2)
        return score

    def compute_score(self, heads, tails, nHeads, nTails, heads_r, tails_r, relations, relations_desc_emb=None):
        heads = heads[:, 0, :].unsqueeze(1)
        tails = tails[:, 0, :].unsqueeze(1)
        heads_r = heads_r[:, 0, :].unsqueeze(1)
        tails_r = tails_r[:, 0, :].unsqueeze(1)

        nHeads = nHeads[:, 0, :].view(heads.size(0), -1, self.config.embedding_size)
        nTails = nTails[:, 0, :].view(tails.size(0), -1, self.config.embedding_size)

        if relations_desc_emb is not None:
            relations = relations_desc_emb[:, 0, :].unsqueeze(1)
        else:
            relations = self.relation_embedding(relations).unsqueeze(1)

        heads = heads.type(torch.FloatTensor)
        tails = tails.type(torch.FloatTensor)
        nHeads = nHeads.type(torch.FloatTensor)
        nTails = nTails.type(torch.FloatTensor)
        heads_r = heads_r.type(torch.FloatTensor)
        tails_r = tails_r.type(torch.FloatTensor)

        relations = relations.type(torch.FloatTensor)

        pScores = (self.score_function(heads_r, relations, tails) + self.score_function(heads, relations, tails_r)) / 2.0
        nHScores = self.score_function(nHeads, relations, tails_r)
        nTScores = self.score_function(heads_r, relations, nTails)
        nScores = torch.cat((nHScores, nTScores), dim=1)
        return pScores, nScores

    def forward(self, heads, tails, nHeads, nTails, heads_r, tails_r, relations, relations_desc, relation_desc_emb=None, **kwargs):
        if relations_desc is not None:
            relation_desc_emb, _ = self.encoder(relations)  # Relations is encoded
        else:
            relation_desc_emb = None # Relation is embedded

        ke_states = {
            'heads': self.encoder(heads)[0],
            'tails': self.encoder(tails)[0],
            'nHeads': self.encoder(nHeads)[0],
            'nTails': self.encoder(nTails)[0],
            'heads_r': self.encoder(heads_r)[0],
            'tails_r': self.encoder(tails_r)[0],
            'relations': relations,
            'relations_desc_emb': relation_desc_emb,
        }

        pScores, nScores = self.compute_score(**ke_states)

        pLoss = F.logsigmoid(pScores).squeeze(dim=1)
        nLoss = F.logsigmoid(-nScores).mean(dim=1)
        ke_loss = (-pLoss.mean()-nLoss.mean())/2.0
        return pScores, nScores, ke_loss


class KeplerModel(PreTrainedModel):
    config_class = KeplerConfig
    base_model_prefix = "kepler"

    def __init__(self, config, encoder, mlm_head):
        super().__init__(config)

        self.encoder = encoder
        self.mlm_head = mlm_head
        self.ke_head = KeplerKnowledgeEmbeddingPredictions(config, self.encoder)
        self.config = config

    def _init_weights(self, module):
        return self.encoder._init_weights(module)

    def get_encoder(self):
        return self.encoder

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.encoder.set_input_embeddings(value)

    def forward(
        self,
        mlm_data,
        ke_data,
        return_dict=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        mlm_output = self.mlm_head(**mlm_data)

        pScore, nScore, ke_loss = None, None, None
        if ke_data is not None:
            pScore, nScore, ke_loss = self.ke_head(**ke_data)
            
        loss = None
        if ke_loss is not None and mlm_output.loss is not None:
            loss = mlm_output.loss + ke_loss

        if not return_dict:
            output = (mlm_output.logits,) + mlm_output.hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return KeplerForPreTrainingOutput(
            loss=loss,
            logits=mlm_output.logits,
            pScore=pScore,
            nScore=nScore,
            hidden_states=mlm_output.hidden_states,
            attentions=mlm_output.attentions,
        )
    