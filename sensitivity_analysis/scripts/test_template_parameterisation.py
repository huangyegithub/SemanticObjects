#!/usr/bin/env python3
"""
Test Script for Template-Based Parameterization

This script demonstrates how to use the new template-based parameterization
system for SMOL sensitivity analysis.

Author: Sensitivity Analysis Tool  
Date: 2025
"""

import sys
import json
from pathlib import Path
from parameter_modifier_template import TemplateParameterModifier

def test_baseline_generation():
    """Test creating a baseline file from template."""
    print("=== Testing Baseline Generation ===")
    
    template_file = "../templates/simulate_onto_template.smol"
    modifier = TemplateParameterModifier()
    
    try:
        baseline_file = modifier.create_baseline_from_template(template_file)
        print(f"✓ Successfully created baseline: {baseline_file}")
        return True
    except Exception as e:
        print(f"✗ Failed to create baseline: {e}")
        return False

def test_parameter_variation():
    """Test creating files with parameter variations."""
    print("\n=== Testing Parameter Variations ===")
    
    template_file = "../templates/simulate_onto_template.smol"
    modifier = TemplateParameterModifier()
    
    test_cases = [
        {
            "params": {"SHALE_SIZE": 25.0},
            "description": "Increased shale size by 25%"
        },
        {
            "params": {"TEMP_FACTOR": 35.0, "BASE_TEMPERATURE": 3.0},
            "description": "Higher temperature gradient and base temp"
        },
        {
            "params": {"DIV_LAYERS": 40, "TOR_LAYERS": 7},
            "description": "More geological layers"
        }
    ]
    
    success_count = 0
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            modified_file = modifier.modify_parameters_from_template(
                template_file,
                test_case["params"],
                test_case["description"]
            )
            print(f"✓ Test {i}: Successfully created {Path(modified_file).name}")
            success_count += 1
        except Exception as e:
            print(f"✗ Test {i}: Failed - {e}")
    
    print(f"\nParameter variation tests: {success_count}/{len(test_cases)} passed")
    return success_count == len(test_cases)

def test_validation():
    """Test parameter validation."""
    print("\n=== Testing Parameter Validation ===")
    
    template_file = "../templates/simulate_onto_template.smol"
    modifier = TemplateParameterModifier()
    
    # Test valid parameters
    try:
        modifier.modify_parameters_from_template(
            template_file,
            {"SHALE_SIZE": 22.0}  # Within valid range
        )
        print("✓ Valid parameter accepted")
        valid_test_passed = True
    except Exception as e:
        print(f"✗ Valid parameter rejected: {e}")
        valid_test_passed = False
    
    # Test invalid parameters
    try:
        modifier.modify_parameters_from_template(
            template_file,
            {"SHALE_SIZE": 1000.0}  # Out of range
        )
        print("✗ Invalid parameter accepted (should have been rejected)")
        invalid_test_passed = False
    except Exception:
        print("✓ Invalid parameter correctly rejected")
        invalid_test_passed = True
    
    return valid_test_passed and invalid_test_passed

def show_generated_files():
    """Show all generated files."""
    print("\n=== Generated Files ===")
    
    input_dir = Path("../data/input")
    if input_dir.exists():
        smol_files = list(input_dir.glob("*.smol"))
        if smol_files:
            print(f"Found {len(smol_files)} generated SMOL files:")
            for f in sorted(smol_files):
                print(f"  - {f.name}")
        else:
            print("No generated SMOL files found")
    else:
        print("Input directory does not exist")

def main():
    """Run all tests."""
    print("Template-Based Parameterization Test Suite")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("../templates/simulate_onto_template.smol").exists():
        print("Error: Template file not found. Please run from sensitivity_analysis/scripts/ directory")
        sys.exit(1)
    
    if not Path("../config/parameters.json").exists():
        print("Error: Parameters config not found. Please run from sensitivity_analysis/scripts/ directory")
        sys.exit(1)
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_baseline_generation():
        tests_passed += 1
    
    if test_parameter_variation():
        tests_passed += 1
    
    if test_validation():
        tests_passed += 1
    
    # Show results
    show_generated_files()
    
    print(f"\n{'=' * 50}")
    print(f"Test Results: {tests_passed}/{total_tests} test suites passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Template parameterization is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())