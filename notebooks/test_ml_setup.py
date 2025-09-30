#!/usr/bin/env python3
"""
Test script to verify ML Pipeline Infrastructure Setup
This script validates that all configurations and imports are working correctly.
"""

import sys
import os

# Add the notebooks directory to the path to import our setup
sys.path.append('notebooks')

def test_ml_setup():
    """Test the ML pipeline setup and configurations."""
    print("Testing ML Pipeline Infrastructure Setup...")
    print("=" * 60)
    
    try:
        # Import the setup script
        exec(open('notebooks/ml_pipeline_setup.py').read())
        print("✓ Setup script executed successfully")
        
        # Test that all required variables are defined
        required_configs = [
            'MODEL_CONFIG', 'FEATURE_CONFIG', 'VALIDATION_CONFIG',
            'IMBALANCE_CONFIG', 'PERSISTENCE_CONFIG', 'REPORT_CONFIG'
        ]
        
        for config in required_configs:
            if config in locals():
                print(f"✓ {config} defined correctly")
            else:
                print(f"✗ {config} not found")
                return False
        
        # Test directory creation
        required_dirs = ['models', 'reports/ml_pipeline', 'visualizations/ml_pipeline']
        for dir_path in required_dirs:
            if os.path.exists(dir_path):
                print(f"✓ Directory '{dir_path}' created successfully")
            else:
                print(f"✗ Directory '{dir_path}' not found")
                return False
        
        # Test random state
        if 'RANDOM_STATE' in locals() and locals()['RANDOM_STATE'] == 42:
            print("✓ Random state set correctly")
        else:
            print("✗ Random state not set correctly")
            return False
        
        print("\n" + "=" * 60)
        print("✓ All tests passed! ML Pipeline Infrastructure is ready.")
        return True
        
    except Exception as e:
        print(f"✗ Error during setup: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_ml_setup()
    sys.exit(0 if success else 1)