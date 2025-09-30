"""
Simple test runner to verify test structure without running pytest
"""

import sys
import os

# Add the production_api directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def check_test_imports():
    """Check if test files can be imported successfully"""
    try:
        print("Checking test imports...")
        
        # Check if we can import the test modules
        from tests import conftest
        print("✓ conftest.py imports successfully")
        
        # Try importing test classes (not running them)
        import ast
        
        test_files = [
            'test_ml_service.py',
            'test_scraper_service.py', 
            'test_classification_service.py',
            'test_api_routes.py'
        ]
        
        for test_file in test_files:
            file_path = os.path.join(os.path.dirname(__file__), test_file)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Parse the file to check for syntax errors
                try:
                    ast.parse(content)
                    print(f"✓ {test_file} has valid syntax")
                except SyntaxError as e:
                    print(f"✗ {test_file} has syntax error: {e}")
            else:
                print(f"✗ {test_file} not found")
        
        print("\nTest structure verification complete!")
        return True
        
    except Exception as e:
        print(f"Error checking imports: {e}")
        return False

if __name__ == "__main__":
    check_test_imports()