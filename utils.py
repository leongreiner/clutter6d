import os
import re
import numpy as np
from scipy.stats import poisson

def create_trunc_poisson_pmf(lambda_param, shift, min_val, max_val):
    # Calculate unnormalized probabilities for all valid values
    values = np.arange(min_val, max_val + 1)
    unnormalized_probs = []
    
    for val in values:
        raw_val = val - shift
        if raw_val >= 0:
            prob = poisson.pmf(raw_val, lambda_param)
        else:
            prob = 0.0
        unnormalized_probs.append(prob)
    
    # Normalize probabilities
    unnormalized_probs = np.array(unnormalized_probs)
    total_prob = np.sum(unnormalized_probs)
    probabilities = unnormalized_probs / total_prob
    
    return values, probabilities

def get_obj_id(model_path):
    filename = os.path.basename(model_path)
    parts = filename.split('__', 1)
    if len(parts) == 2:
        obj_id_part = parts[0]
        obj_id_str = obj_id_part.replace('obj_id_', '')
        obj_id = int(obj_id_str)
        return obj_id

def get_category(model_path):
    filename = os.path.basename(model_path)
    cat_match = re.search(r'__cat_([^_]+(?:_[^_]+)*)__', filename)
    category = cat_match.group(1)
    return category.replace('_', ' ')


def get_name(model_path):
    filename = os.path.basename(model_path)
    name_without_ext = filename.replace('.glb', '').replace('.GLB', '')
    name_match = re.search(r'__cat_[^_]+(?:_[^_]+)*__(.+)', name_without_ext)
    return name_match.group(1)
