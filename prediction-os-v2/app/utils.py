import json
import logging
import random
import numpy as np
import torch

def load_json(file_nm):
    with open(file_nm, 'r') as f:
        jsons = json.load(f)
    return jsons

def create_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)