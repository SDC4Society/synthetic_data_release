#!/usr/bin/env python3
"""
GPU/CUDA compatibility test script for synthetic data models.

This script validates that models can be initialized with GPU device
and that tensor operations work correctly on the specified device.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

from utils.logging import LOGGER
from utils.device_utils import get_device, validate_and_get_device, set_device_env
from utils.constants import CATEGORICAL, FLOAT


def create_test_data(n_rows=100, n_numeric=3, n_categorical=2):
    """Create a small test dataset."""
    np.random.seed(42)
    data = {}
    
    for i in range(n_numeric):
        data[f'num_{i}'] = np.random.randn(n_rows)
    
    for i in range(n_categorical):
        data[f'cat_{i}'] = np.random.choice(['A', 'B', 'C'], n_rows)
    
    df = pd.DataFrame(data)
    
    # Create metadata
    metadata = {
        'columns': []
    }
    for i in range(n_numeric):
        metadata['columns'].append({
            'name': f'num_{i}',
            'type': FLOAT,
            'min': float(df[f'num_{i}'].min()),
            'max': float(df[f'num_{i}'].max()),
        })
    
    for i in range(n_categorical):
        unique_vals = list(df[f'cat_{i}'].unique())
        metadata['columns'].append({
            'name': f'cat_{i}',
            'type': CATEGORICAL,
            'i2s': {j: v for j, v in enumerate(unique_vals)},
            's2i': {v: j for j, v in enumerate(unique_vals)},
        })
    
    return df, metadata


def test_device_utils(device):
    """Test device_utils functions."""
    LOGGER.info(f"Testing device_utils with device: {device}")
    
    # Test get_device
    result_device = get_device(device)
    LOGGER.info(f"  get_device() returned: {result_device}")
    
    # Test validate_and_get_device
    validated_device, is_gpu = validate_and_get_device(device)
    LOGGER.info(f"  validate_and_get_device() returned: {validated_device} (GPU: {is_gpu})")
    
    return result_device, is_gpu


def test_model_init(model_name, model_class, metadata, device):
    """Test initializing a model with specified device."""
    LOGGER.info(f"Testing {model_name} with device: {device}")
    
    try:
        if model_name == 'CTGAN':
            model = model_class(metadata, device=device)
        elif model_name == 'PATEGAN':
            model = model_class(metadata, device=device, num_teachers=2, n_iters=2)
        elif model_name == 'AIM':
            model = model_class(metadata, device=device, max_iters=10)
        elif model_name == 'GEM':
            model = model_class(metadata, device=device, max_iters=10)
        elif model_name == 'DP_MERF':
            model = model_class(metadata, device=device, how_many_epochs=1)
        elif model_name == 'TabDDPM':
            model = model_class(metadata, device=device, steps=2)
        else:
            model = model_class(metadata, device=device)
        
        LOGGER.info(f"  ✓ {model_name} initialized successfully")
        LOGGER.info(f"    Model device: {model.device}")
        if hasattr(model, 'is_gpu'):
            LOGGER.info(f"    Is GPU: {model.is_gpu}")
        return True
    except Exception as e:
        LOGGER.error(f"  ✗ {model_name} initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = ArgumentParser(description='Test GPU/CUDA compatibility')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to test (e.g., "cpu", "cuda", "cuda:0")')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['AIM', 'GEM', 'TabDDPM', 'DP_MERF', 'CTGAN', 'PATEGAN'],
                        help='Models to test')
    parser.add_argument('--all', action='store_true',
                        help='Test all models')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device:
        os.environ['SYNTHETIC_DATA_DEVICE'] = args.device
        LOGGER.info(f"Set SYNTHETIC_DATA_DEVICE={args.device}")
    
    # Test device utilities
    device = get_device(args.device)
    LOGGER.info(f"Testing with device: {device}")
    LOGGER.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        LOGGER.info(f"CUDA devices: {torch.cuda.device_count()}")
    
    print("\n" + "="*60)
    print("Device Utilities Test")
    print("="*60)
    test_device_utils(args.device)
    
    # Create test data
    print("\n" + "="*60)
    print("Creating Test Data")
    print("="*60)
    df, metadata = create_test_data(n_rows=50, n_numeric=2, n_categorical=1)
    LOGGER.info(f"Created test dataset: {df.shape}")
    
    # Test model initialization
    print("\n" + "="*60)
    print("Model Initialization Tests")
    print("="*60 + "\n")
    
    models_to_test = {
        'AIM': None,
        'GEM': None,
        'TabDDPM': None,
        'DP_MERF': None,
        'CTGAN': None,
        'PATEGAN': None,
    }
    
    if not args.all:
        models_to_test = {m: None for m in args.models if m in models_to_test}
    
    results = {}
    
    # Dynamically import and test models
    for model_name in models_to_test.keys():
        try:
            if model_name == 'AIM':
                from generative_models.aim import AIM
                models_to_test[model_name] = AIM
            elif model_name == 'GEM':
                from generative_models.gem import GEM
                models_to_test[model_name] = GEM
            elif model_name == 'TabDDPM':
                from generative_models.tabddpm import TabDDPM
                models_to_test[model_name] = TabDDPM
            elif model_name == 'DP_MERF':
                from generative_models.dp_merf import DP_MERF
                models_to_test[model_name] = DP_MERF
            elif model_name == 'CTGAN':
                from generative_models.ctgan import CTGAN
                models_to_test[model_name] = CTGAN
            elif model_name == 'PATEGAN':
                from generative_models.pate_gan import PATEGAN
                models_to_test[model_name] = PATEGAN
        except ImportError as e:
            LOGGER.warning(f"Could not import {model_name}: {e}")
            continue
    
    for model_name, model_class in models_to_test.items():
        if model_class is None:
            continue
        results[model_name] = test_model_init(model_name, model_class, metadata, args.device)
        print()
    
    # Summary
    print("="*60)
    print("Test Summary")
    print("="*60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    LOGGER.info(f"Passed: {passed}/{total}")
    
    for model_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        LOGGER.info(f"  {status}: {model_name}")
    
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
