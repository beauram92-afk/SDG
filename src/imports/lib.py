# src/imports/lib.py
"""Central place for every library the project uses.
Add new imports to the matching section. Do not import anything from our own project here
(this avoids circular imports)."""
from __future__ import annotations

# ---------- Standard library ----------
import argparse
import copy
import hashlib
import itertools
import json
import logging
import math
import os
import pickle
import random
import re
import sys
import time
import uuid
import warnings
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

# ---------- Data handling ----------
import numpy as np
import pandas as pd

# ---------- Machine learning / statistics ----------
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp
from sklearn.impute import KNNImputer
from sklearn.preprocessing import QuantileTransformer

# ---------- Deep learning ----------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import pytorch_lightning as pl

# ---------- Utilities ----------
from tqdm import tqdm

# ---------- Optional libraries (project still runs if missing) ----------
try:
    from torch_geometric.data import HeteroData
    from torch_geometric.loader import NeighborLoader
    _PYG_AVAILABLE = True
except ImportError:
    HeteroData = None  # type: ignore
    NeighborLoader = None  # type: ignore
    _PYG_AVAILABLE = False

try:
    from faker import Faker
    _FAKER_AVAILABLE = True
except ImportError:  # pragma: no cover
    Faker = None  # type: ignore
    _FAKER_AVAILABLE = False

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

try:
    import optuna
except ImportError:
    optuna = None  # type: ignore