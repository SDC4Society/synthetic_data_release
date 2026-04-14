#!/usr/bin/env python3
"""
Small-scale integration test for GPU-compatible models.

This tests actual fitting and generation with GPU device.
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
from utils.device_utils import get_device, validate_and_get_device
from utils.constants import CATEGORICAL, FLOAT


def create_test_data(n_rows=200, seed=42):
    """Create a test dataset (similar to eval data)."""
    np.random.seed(seed)
    
    data = {
        'age': np.random.randint(18, 80, n_rows),
        'income': np.random.choice(['low', 'medium', 'high'], n_rows),
        'score': np.random.randn(n_rows),
    }
    
    df = pd.DataFrame(data)
    
    # Create metadata
    metadata = {
        'columns': [
            {
                'name': 'age',
                'type': 'Integer',
                'min': int(df['age'].min()),
                'max': int(df['age'].max()),
            },
            {
                'name': 'income',
                'type': 'Categorical',
                'i2s': {0: 'low', 1: 'medium', 2: 'high'},
                's2i': {'low': 0, 'medium': 1, 'high': 2},
            },
            {
                'name': 'score',
                'type': 'Float',
                'min': float(df['score'].min()),
                'max': float(df['score'].max()),
            },
        ]
    }
    
    return df, metadata


def test_model_fit_generate(model_name, model_class, data, metadata, device, n_samples=50):
    """Test fitting and generating samples with a model."""
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"Testing {model_name} (device={device})")
    LOGGER.info('='*60)
    
    try:
        # Initialize model with specified device
        if model_name == 'CTGAN':
            model = model_class(metadata, device=device, epochs=2, batch_size=32)
        elif model_name == 'PATEGAN':
            model = model_class(metadata, device=device, n_iters=2)
        elif model_name == 'AIM':
            model = model_class(metadata, device=device, max_iters=3)
        elif model_name == 'GEM':
            model = model_class(metadata, device=device, max_iters=3)
        elif model_name == 'DP_MERF':
            model = model_class(metadata, device=device, how_many_epochs=1)
        elif model_name == 'TabDDPM':
            # TabDDPM is slower, use minimal params
            model = model_class(metadata, device=device, steps=2, num_timesteps=10)
        else:
            model = model_class(metadata, device=device)
        
        LOGGER.info(f"✓ Model initialized (device={model.device})")
        
        # Fit model
        LOGGER.info(f"Fitting model with {len(data)} samples...")
        model.fit(data)
        LOGGER.info(f"✓ Model fitted successfully")
        
        # Generate samples
        LOGGER.info(f"Generating {n_samples} samples...")
        synthetic_data = model.generate_samples(n_samples)
        LOGGER.info(f"✓ Generated {synthetic_data.shape[0]} synthetic samples")
        
        return True
        
    except Exception as e:
        LOGGER.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = ArgumentParser(description='Test GPU-compatible model fitting and generation')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to test ("cpu" or "cuda:0")')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['AIM', 'GEM', 'DP_MERF'],
                        help='Models to test')
    args = parser.parse_args()
    
    # Set device
    device = get_device(args.device)
    LOGGER.info(f"Testing with device: {device}")
    LOGGER.info(f"CUDA available: {torch.cuda.is_available()}")
    
    # Create test data
    LOGGER.info("Creating test data...")
    data, metadata = create_test_data(n_rows=200)
    LOGGER.info(f"Created dataset: {data.shape}")
    
    # Test models
    results = {}
    
    models_to_test = {
        'AIM': None,
        'GEM': None,
        'DP_MERF': None,
        'CTGAN': None,
        'PATEGAN': None,
        'TabDDPM': None,
    }
    
    # Filter to requested models
    models_to_test = {m: None for m in args.models if m in models_to_test}
    
    # Dynamically import models
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
    
    # Run tests
    for model_name, model_class in models_to_test.items():
        if model_class is None:
            continue
        results[model_name] = test_model_fit_generate(
            model_name, model_class, data, metadata, args.device
        )
    
    # Summary
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info("Test Summary")
    LOGGER.info('='*60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    LOGGER.info(f"Passed: {passed}/{total}")
    
    for model_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        LOGGER.info(f"  {status}: {model_name}")
    
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
