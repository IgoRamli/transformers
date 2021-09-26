""" KEPLER Knowledge Embedding model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

KEPLER_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class KeplerConfig(PretrainedConfig):
    model_type = "kepler"

    def __init__(
        self,
        embedding_size=768,
        nrelation=1000,
        gamma=4,
        ke_model='TransE',
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.embedding_size = embedding_size  # Must be equal to size of last hidden state
        self.nrelation = nrelation  # Number of distinct relations
        self.gamma = gamma
        self.ke_model = ke_model