import torch
import torch.nn as nn
from models import model_map
from models.utils import Config, Int8QuantHandler, WeightOnlyInt4QuantHandler, get_tokenizer, \
    sample, norm_logits, norm_max
from typing import Optional
from torch import Tensor
from pathlib import Path
from transformers import AutoTokenizer
from torch.distributed.distributed_c10d import is_torchelastic_launched

default_devices = ['cuda:1', 'cuda:2', 'cuda:3']

