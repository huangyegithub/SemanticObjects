#!/usr/bin/env python3
"""
Parameter extraction utility for SMOL geological simulation files.
Extracts parameter definitions and creates configuration files.
"""

import re
import json
from typing import Dict, List, Tuple, Any
from pathlib import Path


class ParameterExtractor:
    """Extracts parameters from SMOL files and creates configuration files."""
    
    def __init__(self, smol_file_path: str):
        self.smol_file_path = Path(smol_file_path)
        self.parameters = {}
        
    def extract_parameters(self) -> Dict[str, Any]:
        """Extract parameter definitions from SMOL file."""
        parameter_pattern = r'^(\w+)\s*=\s*([0-9.-]+)\s*//\s*(.+)$'
        
        with open(self.smol_file_path, 'r') as f:
            content = f.read()
        
        # Find parameter definitions in the header section
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            match = re.match(parameter_pattern, line)
            
            if match:
                param_name = match.group(1)
                param_value = float(match.group(2))
                param_description = match.group(3).strip()
                
                self.parameters[param_name] = {
                    'value': param_value,
                    'description': param_description,
                    'line_number': line_num,
                    'category': self._categorize_parameter(param_name),
                    'units': self._extract_units(param_description),
                    'type': 'float' if '.' in match.group(2) else 'int'
                }
        
        return self.parameters
    
    def _categorize_parameter(self, param_name: str) -> str:
        """Categorize parameter based on its name."""
        if 'SIZE' in param_name:
            return 'geological_size'
        elif 'TEMP' in param_name:
            return 'thermal'
        elif 'LAYERS' in param_name:
            return 'structural'
        elif param_name in ['START_PAST', 'CHECK_START', 'DEPOSITION_DURATION']:
            return 'temporal'
        elif param_name == 'HYDROCARBON_INCREMENT':
            return 'simulation'
        else:
            return 'other'
    
    def _extract_units(self, description: str) -> str:
        """Extract units from parameter description."""
        units_pattern = r'\(([^)]+)\)'
        match = re.search(units_pattern, description)
        if match:
            return match.group(1)
        
        # Common unit patterns
        if 'meters' in description.lower():
            return 'meters'
        elif 'celsius' in description.lower():
            return 'celsius'
        elif 'years' in description.lower():
            return 'million years'
        elif 'layers' in description.lower():
            return 'count'
        else:
            return 'dimensionless'
    
    def generate_parameter_ranges(self) -> Dict[str, Any]:
        """Generate reasonable parameter variation ranges."""
        ranges = {}
        
        for param_name, param_info in self.parameters.items():
            base_value = param_info['value']
            category = param_info['category']
            
            # Define variation percentages by category
            if category == 'geological_size':
                variation = 0.3  # ±30% for geological sizes
            elif category == 'thermal':
                variation = 0.5  # ±50% for temperature parameters
            elif category == 'temporal':
                variation = 0.2  # ±20% for time parameters
            elif category == 'structural':
                variation = 0.4  # ±40% for layer counts (but ensure integers)
            else:
                variation = 0.3  # Default ±30%
            
            # Calculate min/max values
            if base_value >= 0:
                min_val = base_value * (1 - variation)
                max_val = base_value * (1 + variation)
            else:
                # For negative values, flip the logic
                min_val = base_value * (1 + variation)
                max_val = base_value * (1 - variation)
            
            # For integer parameters, ensure integer ranges
            if param_info['type'] == 'int':
                min_val = max(1, int(min_val))  # Ensure at least 1
                max_val = max(min_val + 1, int(max_val))  # Ensure max > min
            
            ranges[param_name] = {
                'min': min_val,
                'max': max_val,
                'base': base_value,
                'variation_percent': variation * 100,
                'type': param_info['type'],
                'category': category,
                'units': param_info['units'],
                'description': param_info['description']
            }
        
        return ranges
    
    def create_config_files(self, config_dir: str) -> None:
        """Create configuration files in the specified directory."""
        config_path = Path(config_dir)
        config_path.mkdir(exist_ok=True)
        
        # Extract parameters
        parameters = self.extract_parameters()
        
        # Generate parameter ranges
        parameter_ranges = self.generate_parameter_ranges()
        
        # Create parameters.json
        parameters_config = {
            'metadata': {
                'source_file': str(self.smol_file_path),
                'total_parameters': len(parameters),
                'extraction_date': self._get_current_date()
            },
            'parameters': parameter_ranges
        }
        
        with open(config_path / 'parameters.json', 'w') as f:
            json.dump(parameters_config, f, indent=2)
        
        # Create baseline.json
        baseline_config = {
            'metadata': {
                'source_file': str(self.smol_file_path),
                'description': 'Baseline parameter values from original SMOL file'
            },
            'baseline_values': {
                name: info['value'] for name, info in parameters.items()
            }
        }
        
        with open(config_path / 'baseline.json', 'w') as f:
            json.dump(baseline_config, f, indent=2)
        
        print(f"Created configuration files in {config_path}")
        print(f"- parameters.json: {len(parameters)} parameters with ranges")
        print(f"- baseline.json: baseline values")
    
    def _get_current_date(self) -> str:
        """Get current date as string."""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def main():
    """Main function for command-line usage."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python parameter_extractor.py <smol_file_path>")
        sys.exit(1)
    
    smol_file = sys.argv[1]
    extractor = ParameterExtractor(smol_file)
    
    # Extract parameters
    parameters = extractor.extract_parameters()
    
    print(f"Extracted {len(parameters)} parameters:")
    for name, info in parameters.items():
        print(f"  {name}: {info['value']} ({info['units']}) - {info['description']}")
    
    # Create config files
    extractor.create_config_files('./config')


if __name__ == '__main__':
    main()