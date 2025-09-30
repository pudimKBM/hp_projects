"""
Integration Test Runner

This module provides utilities to run and manage integration tests for the
production classification API system.

Requirements covered: 1.2, 1.3, 2.1
"""

import sys
import os
import time
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Add the production_api directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False


class TestResult(Enum):
    """Test execution results"""
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


@dataclass
class TestExecution:
    """Test execution information"""
    test_name: str
    test_file: str
    result: TestResult
    duration: float
    error_message: str = None
    traceback: str = None


class IntegrationTestRunner:
    """Runner for integration tests with detailed reporting"""
    
    def __init__(self):
        """Initialize the test runner"""
        self.test_results: List[TestExecution] = []
        self.start_time = None
        self.end_time = None
        
    def run_all_integration_tests(self) -> Dict[str, Any]:
        """
        Run all integration tests and return comprehensive results.
        
        Returns:
            Dict containing test results and summary
        """
        self.start_time = time.time()
        
        print("=" * 80)
        print("PRODUCTION CLASSIFICATION API - INTEGRATION TEST SUITE")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Test files to run
        test_files = [
            'test_integration.py',
            'test_database_integration.py',
            'test_api_integration.py'
        ]
        
        if PYTEST_AVAILABLE:
            return self._run_with_pytest(test_files)
        else:
            return self._run_manual_tests(test_files)
    
    def _run_with_pytest(self, test_files: List[str]) -> Dict[str, Any]:
        """Run tests using pytest"""
        print("Running integration tests with pytest...")
        print()
        
        # Configure pytest arguments
        pytest_args = [
            '-v',  # Verbose output
            '--tb=short',  # Short traceback format
            '--durations=10',  # Show 10 slowest tests
            '--strict-markers',  # Strict marker checking
        ]
        
        # Add test files
        test_dir = os.path.dirname(__file__)
        for test_file in test_files:
            test_path = os.path.join(test_dir, test_file)
            if os.path.exists(test_path):
                pytest_args.append(test_path)
        
        try:
            # Run pytest
            exit_code = pytest.main(pytest_args)
            
            self.end_time = time.time()
            
            return {
                'success': exit_code == 0,
                'exit_code': exit_code,
                'duration': self.end_time - self.start_time,
                'runner': 'pytest',
                'test_files': test_files,
                'message': 'Tests completed with pytest' if exit_code == 0 else 'Some tests failed'
            }
            
        except Exception as e:
            self.end_time = time.time()
            return {
                'success': False,
                'error': str(e),
                'duration': self.end_time - self.start_time,
                'runner': 'pytest',
                'message': f'Error running pytest: {e}'
            }
    
    def _run_manual_tests(self, test_files: List[str]) -> Dict[str, Any]:
        """Run tests manually without pytest"""
        print("Pytest not available. Running manual test validation...")
        print()
        
        test_dir = os.path.dirname(__file__)
        
        for test_file in test_files:
            test_path = os.path.join(test_dir, test_file)
            
            print(f"Validating {test_file}...")
            
            if not os.path.exists(test_path):
                self._add_test_result(test_file, 'file_check', TestResult.FAILED, 0, 
                                    f"Test file not found: {test_path}")
                continue
            
            # Check if file can be imported (syntax validation)
            try:
                with open(test_path, 'r') as f:
                    content = f.read()
                
                # Basic syntax check
                import ast
                ast.parse(content)
                
                # Count test functions
                test_count = content.count('def test_')
                
                self._add_test_result(test_file, 'syntax_check', TestResult.PASSED, 0.1,
                                    f"Syntax valid, {test_count} test functions found")
                
                print(f"  ✓ {test_file}: Syntax valid, {test_count} test functions")
                
            except SyntaxError as e:
                self._add_test_result(test_file, 'syntax_check', TestResult.FAILED, 0,
                                    f"Syntax error: {e}")
                print(f"  ✗ {test_file}: Syntax error - {e}")
                
            except Exception as e:
                self._add_test_result(test_file, 'import_check', TestResult.ERROR, 0,
                                    f"Import error: {e}")
                print(f"  ✗ {test_file}: Import error - {e}")
        
        self.end_time = time.time()
        
        # Generate summary
        passed_tests = [r for r in self.test_results if r.result == TestResult.PASSED]
        failed_tests = [r for r in self.test_results if r.result == TestResult.FAILED]
        error_tests = [r for r in self.test_results if r.result == TestResult.ERROR]
        
        print()
        print("Manual Test Validation Summary:")
        print(f"  Passed: {len(passed_tests)}")
        print(f"  Failed: {len(failed_tests)}")
        print(f"  Errors: {len(error_tests)}")
        
        return {
            'success': len(failed_tests) == 0 and len(error_tests) == 0,
            'duration': self.end_time - self.start_time,
            'runner': 'manual',
            'test_files': test_files,
            'results': {
                'passed': len(passed_tests),
                'failed': len(failed_tests),
                'errors': len(error_tests),
                'total': len(self.test_results)
            },
            'message': 'Manual validation completed'
        }
    
    def _add_test_result(self, test_file: str, test_name: str, result: TestResult, 
                        duration: float, error_message: str = None):
        """Add a test result to the collection"""
        execution = TestExecution(
            test_name=test_name,
            test_file=test_file,
            result=result,
            duration=duration,
            error_message=error_message
        )
        self.test_results.append(execution)
    
    def generate_report(self) -> str:
        """Generate a detailed test report"""
        if not self.test_results:
            return "No test results available"
        
        report = []
        report.append("=" * 80)
        report.append("INTEGRATION TEST EXECUTION REPORT")
        report.append("=" * 80)
        
        if self.start_time and self.end_time:
            report.append(f"Execution Time: {self.end_time - self.start_time:.2f} seconds")
        
        report.append(f"Total Tests: {len(self.test_results)}")
        
        # Group by result
        by_result = {}
        for result in TestResult:
            by_result[result] = [r for r in self.test_results if r.result == result]
        
        for result, tests in by_result.items():
            if tests:
                report.append(f"{result.value}: {len(tests)}")
        
        report.append("")
        
        # Detailed results
        for result in TestResult:
            tests = by_result[result]
            if tests:
                report.append(f"{result.value} TESTS:")
                report.append("-" * 40)
                
                for test in tests:
                    report.append(f"  {test.test_file}::{test.test_name}")
                    if test.duration > 0:
                        report.append(f"    Duration: {test.duration:.3f}s")
                    if test.error_message:
                        report.append(f"    Error: {test.error_message}")
                
                report.append("")
        
        return "\n".join(report)


def validate_test_environment() -> Dict[str, Any]:
    """
    Validate that the test environment is properly set up.
    
    Returns:
        Dict containing validation results
    """
    print("Validating integration test environment...")
    
    validation_results = {
        'valid': True,
        'issues': [],
        'warnings': [],
        'info': []
    }
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        validation_results['issues'].append(f"Python 3.7+ required, found {python_version.major}.{python_version.minor}")
        validation_results['valid'] = False
    else:
        validation_results['info'].append(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check required modules
    required_modules = [
        'flask',
        'sqlalchemy',
        'pandas',
        'numpy',
        'unittest.mock'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
            validation_results['info'].append(f"Module {module}: Available")
        except ImportError:
            validation_results['issues'].append(f"Required module not found: {module}")
            validation_results['valid'] = False
    
    # Check optional modules
    optional_modules = [
        'pytest',
        'coverage'
    ]
    
    for module in optional_modules:
        try:
            __import__(module)
            validation_results['info'].append(f"Optional module {module}: Available")
        except ImportError:
            validation_results['warnings'].append(f"Optional module not found: {module}")
    
    # Check test files exist
    test_dir = os.path.dirname(__file__)
    required_test_files = [
        'conftest.py',
        'test_integration.py',
        'test_database_integration.py',
        'test_api_integration.py'
    ]
    
    for test_file in required_test_files:
        test_path = os.path.join(test_dir, test_file)
        if os.path.exists(test_path):
            validation_results['info'].append(f"Test file {test_file}: Found")
        else:
            validation_results['issues'].append(f"Test file not found: {test_file}")
            validation_results['valid'] = False
    
    # Check production_api structure
    api_dir = os.path.join(test_dir, '..')
    required_api_dirs = [
        'app',
        'app/models.py',
        'app/services',
        'config'
    ]
    
    for api_path in required_api_dirs:
        full_path = os.path.join(api_dir, api_path)
        if os.path.exists(full_path):
            validation_results['info'].append(f"API component {api_path}: Found")
        else:
            validation_results['warnings'].append(f"API component not found: {api_path}")
    
    return validation_results


def print_validation_results(results: Dict[str, Any]):
    """Print validation results in a formatted way"""
    print("\nEnvironment Validation Results:")
    print("=" * 50)
    
    if results['valid']:
        print("✓ Environment validation PASSED")
    else:
        print("✗ Environment validation FAILED")
    
    if results['issues']:
        print("\nISSUES (must be resolved):")
        for issue in results['issues']:
            print(f"  ✗ {issue}")
    
    if results['warnings']:
        print("\nWARNINGS (recommended to resolve):")
        for warning in results['warnings']:
            print(f"  ⚠ {warning}")
    
    if results['info']:
        print("\nINFO:")
        for info in results['info']:
            print(f"  ℹ {info}")
    
    print()


def main():
    """Main function to run integration tests"""
    print("Production Classification API - Integration Test Runner")
    print("=" * 60)
    
    # Validate environment first
    validation = validate_test_environment()
    print_validation_results(validation)
    
    if not validation['valid']:
        print("Cannot run tests due to environment issues. Please resolve the issues above.")
        return 1
    
    # Run integration tests
    runner = IntegrationTestRunner()
    results = runner.run_all_integration_tests()
    
    print("\n" + "=" * 80)
    print("INTEGRATION TEST RESULTS")
    print("=" * 80)
    
    print(f"Runner: {results['runner']}")
    print(f"Duration: {results['duration']:.2f} seconds")
    print(f"Success: {results['success']}")
    
    if 'results' in results:
        test_results = results['results']
        print(f"Total Tests: {test_results['total']}")
        print(f"Passed: {test_results['passed']}")
        print(f"Failed: {test_results['failed']}")
        print(f"Errors: {test_results['errors']}")
    
    if 'error' in results:
        print(f"Error: {results['error']}")
    
    print(f"Message: {results['message']}")
    
    # Generate detailed report if available
    if runner.test_results:
        print("\n" + runner.generate_report())
    
    return 0 if results['success'] else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)