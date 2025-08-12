#!/usr/bin/env python3
"""
Enhanced Parameter Modifier Module for SMOL Sensitivity Analysis
Supports both regex-based and template-based parameter substitution

This module provides functionality to modify SMOL simulation parameters
using template placeholders ({{PARAMETER_NAME}}) for cleaner parameter replacement.

Author: Sensitivity Analysis Tool
Date: 2024
Enhanced: 2025
"""

import re
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union
import hashlib
from datetime import datetime


class TemplateParameterModifier:
    """
    Modifies SMOL file parameters using template placeholder substitution.
    
    This enhanced version supports:
    - Template-based substitution using {{PARAMETER_NAME}} placeholders
    - Original regex-based substitution for backward compatibility
    - SimConfig class parameter management
    - Generating unique filenames for modified files
    - Preserving file structure and comments
    """
    
    def __init__(self, config_dir: str = "../config", output_dir: str = "../data/input"):
        """
        Initialize the parameter modifier.
        
        Args:
            config_dir: Directory containing parameter configuration files
            output_dir: Directory where modified SMOL files will be saved
        """
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load parameter configuration
        self._load_config()
        
    def _load_config(self) -> None:
        """Load parameter configuration from JSON files."""
        params_file = self.config_dir / "parameters.json"
        baseline_file = self.config_dir / "baseline.json"
        print(params_file, baseline_file)

        with open(params_file, 'r') as f:
            self.params_config = json.load(f)
            
        # Create baseline config if it doesn't exist
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                self.baseline_config = json.load(f)
        else:
            # Generate baseline config from parameters
            self.baseline_config = {
                "baseline_values": {
                    param_name: param_info["base"]
                    for param_name, param_info in self.params_config["parameters"].items()
                }
            }
            # Save baseline config for future use
            with open(baseline_file, 'w') as f:
                json.dump(self.baseline_config, f, indent=2)
    
    def modify_parameters_from_template(self, 
                                      template_file: str, 
                                      param_values: Optional[Dict[str, Union[float, int]]] = None,
                                      description: Optional[str] = None) -> str:
        """
        Create a modified SMOL file from a template with {{PARAMETER}} placeholders.
        
        Args:
            template_file: Path to the template SMOL file with placeholders
            param_values: Dictionary of parameter names and their values
                         If None, uses baseline values
            description: Optional description of the modification
            
        Returns:
            Path to the generated modified SMOL file
            
        Example:
            >>> modifier = TemplateParameterModifier()
            >>> new_file = modifier.modify_parameters_from_template(
            ...     "simulate_onto_template.smol",
            ...     {"SHALE_SIZE": 25.0, "TEMP_FACTOR": 35.0},
            ...     "Increased shale size and temperature"
            ... )
        """
        template_path = Path(template_file)
        
        # Use baseline values if no parameters specified
        if param_values is None:
            param_values = self.baseline_config['baseline_values'].copy()
        else:
            # Merge with baseline to ensure all parameters have values
            all_params = self.baseline_config['baseline_values'].copy()
            all_params.update(param_values)
            param_values = all_params
        
        # Validate parameters
        self._validate_parameters(param_values)
        
        # Read template file
        with open(template_path, 'r') as f:
            content = f.read()
        
        # Generate unique filename based on parameter changes
        param_changes = self._get_parameter_changes(param_values)
        output_filename = self._generate_filename(template_path, param_changes)
        output_path = self.output_dir / output_filename
        
        # Apply template substitution
        modified_content = self._apply_template_substitution(content, param_values)
        
        # Add metadata header
        header = self._generate_metadata_header(template_file, param_changes, description)
        modified_content = header + modified_content
        
        # Write modified file
        with open(output_path, 'w') as f:
            f.write(modified_content)
        
        print(f"Created modified SMOL file: {output_path}")
        return str(output_path)
    
    def _get_parameter_changes(self, param_values: Dict[str, Union[float, int]]) -> Dict[str, Union[float, int]]:
        """
        Calculate which parameters have changed from baseline.
        
        Args:
            param_values: All parameter values
            
        Returns:
            Dictionary of only the changed parameters
        """
        changes = {}
        baseline = self.baseline_config['baseline_values']
        
        for param_name, value in param_values.items():
            if param_name in baseline and value != baseline[param_name]:
                changes[param_name] = value
                
        return changes
    
    def _apply_template_substitution(self, content: str, param_values: Dict[str, Union[float, int]]) -> str:
        """
        Apply template substitution for {{PARAMETER_NAME}} placeholders.
        
        Args:
            content: Template content with placeholders
            param_values: Dictionary of all parameter values
            
        Returns:
            Content with placeholders replaced by values
        """
        modified_content = content
        
        for param_name, value in param_values.items():
            # Format value based on type
            param_info = self.params_config['parameters'].get(param_name, {})
            param_type = param_info.get('type', 'float')
            
            if param_type == 'int':
                formatted_value = str(int(value))
            else:
                # Use appropriate decimal places
                if value % 1 == 0:
                    formatted_value = f"{value:.1f}"
                else:
                    formatted_value = f"{value:.2f}"
            
            # Replace placeholder
            placeholder = f"{{{{{param_name}}}}}"
            modified_content = modified_content.replace(placeholder, formatted_value)
        
        # Check for any remaining placeholders
        remaining_placeholders = re.findall(r'\{\{([A-Z_]+)\}\}', modified_content)
        if remaining_placeholders:
            print(f"Warning: Unsubstituted placeholders found: {remaining_placeholders}")
        
        return modified_content
    
    def _validate_parameters(self, param_values: Dict[str, Union[float, int]]) -> None:
        """
        Validate that parameter values are within acceptable ranges.
        
        Args:
            param_values: Dictionary of parameter values to validate
            
        Raises:
            ValueError: If any parameter is invalid or out of range
        """
        params_info = self.params_config['parameters']
        
        for param_name, value in param_values.items():
            if param_name not in params_info:
                print(f"Warning: Unknown parameter {param_name}, skipping validation")
                continue
            
            param_config = params_info[param_name]
            min_val = param_config['min']
            max_val = param_config['max']
            
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"Parameter {param_name} value {value} out of range "
                    f"[{min_val}, {max_val}]"
                )
            
            # Ensure integer parameters remain integers
            if param_config['type'] == 'int' and not isinstance(value, int):
                if value % 1 == 0:
                    # Auto-convert float to int if it's a whole number
                    param_values[param_name] = int(value)
                else:
                    raise ValueError(
                        f"Parameter {param_name} must be an integer, got {value}"
                    )
    
    def _generate_filename(self, source_path: Path, param_changes: Dict[str, Union[float, int]]) -> str:
        """
        Generate a unique filename for the modified SMOL file.
        
        Args:
            source_path: Original file path
            param_changes: Dictionary of parameter changes
            
        Returns:
            Unique filename incorporating parameter information
        """
        # Create a hash of parameter changes for uniqueness
        if param_changes:
            param_str = "_".join(f"{k}={v}" for k, v in sorted(param_changes.items()))
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        else:
            param_hash = "baseline"
        
        # Build filename with key parameter info
        base_name = source_path.stem.replace("_template", "")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Include most significant parameter change in filename
        if param_changes:
            # Choose the parameter with the largest relative change
            max_change_param = None
            max_change_pct = 0
            
            for param_name, new_value in param_changes.items():
                baseline_value = self.baseline_config['baseline_values'].get(param_name, new_value)
                if baseline_value != 0:
                    change_pct = abs((new_value - baseline_value) / baseline_value)
                    if change_pct > max_change_pct:
                        max_change_pct = change_pct
                        max_change_param = param_name
            
            if max_change_param:
                key_value = param_changes[max_change_param]
                filename = f"{base_name}_{max_change_param}_{key_value}_{param_hash}.smol"
            else:
                filename = f"{base_name}_modified_{param_hash}.smol"
        else:
            filename = f"{base_name}_baseline_{timestamp}.smol"
        
        return filename
    
    def _generate_metadata_header(self, 
                                 source_file: str, 
                                 param_changes: Dict[str, Union[float, int]],
                                 description: Optional[str] = None) -> str:
        """
        Generate metadata header for the modified SMOL file.
        
        Args:
            source_file: Original file path
            param_changes: Dictionary of parameter changes
            description: Optional description
            
        Returns:
            Formatted metadata header as a comment block
        """
        header_lines = [
            "// ============================================",
            "// GENERATED BY SENSITIVITY ANALYSIS TOOL",
            f"// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"// Source: {source_file}",
            "// ============================================",
        ]
        
        if param_changes:
            header_lines.append("// Modified Parameters:")
            # Sort by parameter name for consistent output
            for param_name in sorted(param_changes.keys()):
                new_value = param_changes[param_name]
                baseline_value = self.baseline_config['baseline_values'].get(param_name, new_value)
                if baseline_value != new_value and baseline_value != 0:
                    change_pct = ((new_value - baseline_value) / baseline_value) * 100
                    header_lines.append(
                        f"// - {param_name}: {baseline_value} -> {new_value} ({change_pct:+.1f}%)"
                    )
                else:
                    header_lines.append(
                        f"// - {param_name}: {new_value}"
                    )
        else:
            header_lines.append("// Configuration: BASELINE (no parameter changes)")
        
        # Add description if provided
        if description:
            header_lines.extend([
                "// Description:",
                f"// {description}"
            ])
        
        header_lines.extend([
            "// ============================================",
            ""  # Empty line before content
        ])
        
        return "\n".join(header_lines) + "\n"
    
    def create_baseline_from_template(self, template_file: str) -> str:
        """
        Create a baseline file from template using baseline parameter values.
        
        Args:
            template_file: Path to the template SMOL file
            
        Returns:
            Path to the baseline file
        """
        return self.modify_parameters_from_template(
            template_file,
            None,  # Use baseline values
            "Baseline configuration with all default parameter values"
        )
    
    # Backward compatibility method
    def modify_parameters(self, source_file: str, param_changes: Dict[str, float], 
                         description: Optional[str] = None) -> str:
        """
        Legacy method for regex-based parameter modification.
        Maintained for backward compatibility.
        """
        print("Note: Using legacy regex-based modification. Consider using template-based approach.")
        # Delegate to parent class implementation
        from parameter_modifier import ParameterModifier
        legacy_modifier = ParameterModifier(self.config_dir, self.output_dir)
        return legacy_modifier.modify_parameters(source_file, param_changes, description)


def main():
    """
    Main function for command-line usage.
    
    Example usage:
        # From template
        python parameter_modifier_template.py -t simulate_onto_template.smol SHALE_SIZE=25.0 TEMP_FACTOR=35.0
        
        # Create baseline
        python parameter_modifier_template.py -t simulate_onto_template.smol --baseline
    """
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Modify SMOL simulation parameters')
    parser.add_argument('-t', '--template', help='Use template file with {{PARAM}} placeholders')
    parser.add_argument('-s', '--source', help='Use source file with regex replacement (legacy)')
    parser.add_argument('--baseline', action='store_true', help='Create baseline configuration')
    parser.add_argument('params', nargs='*', help='Parameter changes as PARAM=VALUE')
    
    args = parser.parse_args()
    
    if not (args.template or args.source):
        print("Error: Must specify either --template or --source file")
        parser.print_help()
        sys.exit(1)
    
    # Parse parameter changes
    param_changes = {}
    for param_arg in args.params:
        if '=' in param_arg:
            param_name, value = param_arg.split('=', 1)
            try:
                # Try to parse as float, then int
                param_value = float(value)
                if param_value.is_integer():
                    param_value = int(param_value)
            except ValueError:
                print(f"Error: Invalid value for {param_name}: {value}")
                sys.exit(1)
            param_changes[param_name] = param_value
    
    # Create modifier
    modifier = TemplateParameterModifier()
    
    try:
        if args.template:
            if args.baseline:
                modified_file = modifier.create_baseline_from_template(args.template)
            else:
                modified_file = modifier.modify_parameters_from_template(
                    args.template, 
                    param_changes if param_changes else None
                )
        else:
            # Legacy mode
            modified_file = modifier.modify_parameters(args.source, param_changes)
            
        print(f"Success! Modified file created: {modified_file}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()