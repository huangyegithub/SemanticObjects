#!/usr/bin/env python3
"""
Parameter Modifier Module for SMOL Sensitivity Analysis

This module provides functionality to modify SMOL simulation parameters
and generate new SMOL files with updated parameter values. Modified files
are saved to a separate directory to preserve the original.

Author: Sensitivity Analysis Tool
Date: 2024
"""

import re
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib
from datetime import datetime


class ParameterModifier:
    """
    Modifies SMOL file parameters and generates new simulation files.
    
    This class handles:
    - Reading original SMOL files
    - Modifying specific parameter values
    - Generating unique filenames for modified files
    - Preserving file structure and comments
    """
    
    def __init__(self, config_dir: str = "./config", output_dir: str = "./data/input"):
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
        
        with open(params_file, 'r') as f:
            self.params_config = json.load(f)
            
        with open(baseline_file, 'r') as f:
            self.baseline_config = json.load(f)
    
    def modify_parameters(self, 
                         source_file: str, 
                         param_changes: Dict[str, float],
                         description: Optional[str] = None) -> str:
        """
        Create a modified SMOL file with updated parameter values.
        
        Args:
            source_file: Path to the original SMOL file
            param_changes: Dictionary of parameter names and their new values
            description: Optional description of the modification
            
        Returns:
            Path to the generated modified SMOL file
            
        Example:
            >>> modifier = ParameterModifier()
            >>> new_file = modifier.modify_parameters(
            ...     "simulate.smol",
            ...     {"SHALE_SIZE": 25.0, "TEMP_FACTOR": 35.0},
            ...     "Increased shale size and temperature"
            ... )
        """
        source_path = Path(source_file)
        
        # Validate parameters
        self._validate_parameters(param_changes)
        
        # Read original file
        with open(source_path, 'r') as f:
            content = f.read()
        
        # Generate unique filename based on parameter changes
        output_filename = self._generate_filename(source_path, param_changes)
        output_path = self.output_dir / output_filename
        
        # Apply parameter modifications
        modified_content = self._apply_modifications(content, param_changes)
        
        # Add metadata header
        header = self._generate_metadata_header(source_file, param_changes, description)
        modified_content = header + modified_content
        
        # Write modified file
        with open(output_path, 'w') as f:
            f.write(modified_content)
        
        print(f"Created modified SMOL file: {output_path}")
        return str(output_path)
    
    def _validate_parameters(self, param_changes: Dict[str, float]) -> None:
        """
        Validate that parameter changes are within acceptable ranges.
        
        Args:
            param_changes: Dictionary of parameter changes to validate
            
        Raises:
            ValueError: If any parameter is invalid or out of range
        """
        params_info = self.params_config['parameters']
        
        for param_name, new_value in param_changes.items():
            if param_name not in params_info:
                raise ValueError(f"Unknown parameter: {param_name}")
            
            param_config = params_info[param_name]
            min_val = param_config['min']
            max_val = param_config['max']
            
            if not (min_val <= new_value <= max_val):
                raise ValueError(
                    f"Parameter {param_name} value {new_value} out of range "
                    f"[{min_val}, {max_val}]"
                )
            
            # Ensure integer parameters remain integers
            if param_config['type'] == 'int' and not isinstance(new_value, int):
                raise ValueError(
                    f"Parameter {param_name} must be an integer, got {new_value}"
                )
    
    def _apply_modifications(self, content: str, param_changes: Dict[str, float]) -> str:
        """
        Apply parameter modifications to the SMOL file content.
        
        Args:
            content: Original SMOL file content
            param_changes: Dictionary of parameter changes
            
        Returns:
            Modified content with updated parameter values
        """
        modified_content = content
        
        # === ENHANCEMENT START: Added support for template placeholders ===
        # Check if content uses template placeholders {{PARAM_NAME}}
        has_placeholders = bool(re.search(r'\{\{[A-Z_]+\}\}', content))
        
        if has_placeholders:
            # Use template-based substitution for files with placeholders
            for param_name, new_value in param_changes.items():
                # Format value based on type
                param_type = self.params_config['parameters'][param_name]['type']
                if param_type == 'int':
                    formatted_value = str(int(new_value))
                else:
                    formatted_value = f"{new_value:.1f}" if new_value % 1 == 0 else f"{new_value:.2f}"
                
                # Replace placeholder
                placeholder = f"{{{{{param_name}}}}}"
                modified_content = modified_content.replace(placeholder, formatted_value)
        else:
            # Use original regex-based substitution for backward compatibility
            # === ORIGINAL CODE BELOW (unchanged) ===
            for param_name, new_value in param_changes.items():
                # Pattern to match parameter definition
                # Handles: PARAM_NAME = value // comment
                pattern = rf'^({param_name}\s*=\s*)[0-9.-]+(\s*//.*)$'
                
                # Format value based on type
                param_type = self.params_config['parameters'][param_name]['type']
                if param_type == 'int':
                    formatted_value = str(int(new_value))
                else:
                    formatted_value = f"{new_value:.1f}" if new_value % 1 == 0 else f"{new_value:.2f}"
                
                # Replace parameter value
                replacement = rf'\g<1>{formatted_value}\g<2>'
                modified_content = re.sub(pattern, replacement, modified_content, flags=re.MULTILINE)
        # === ENHANCEMENT END ===
        
        return modified_content
    
    def _generate_filename(self, source_path: Path, param_changes: Dict[str, float]) -> str:
        """
        Generate a unique filename for the modified SMOL file.
        
        Args:
            source_path: Original file path
            param_changes: Dictionary of parameter changes
            
        Returns:
            Unique filename incorporating parameter information
        """
        # Create a hash of parameter changes for uniqueness
        param_str = "_".join(f"{k}={v}" for k, v in sorted(param_changes.items()))
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        
        # Build filename with key parameter info
        base_name = source_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Include most significant parameter change in filename
        if param_changes:
            key_param = list(param_changes.keys())[0]
            key_value = param_changes[key_param]
            filename = f"{base_name}_{key_param}_{key_value}_{param_hash}.smol"
        else:
            filename = f"{base_name}_baseline_{timestamp}.smol"
        
        return filename
    
    def _generate_metadata_header(self, 
                                 source_file: str, 
                                 param_changes: Dict[str, float],
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
            "// Modified Parameters:"
        ]
        
        # Add parameter changes
        for param_name, new_value in param_changes.items():
            baseline_value = self.baseline_config['baseline_values'][param_name]
            change_pct = ((new_value - baseline_value) / baseline_value) * 100
            header_lines.append(
                f"// - {param_name}: {baseline_value} -> {new_value} ({change_pct:+.1f}%)"
            )
        
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
    
    def create_baseline_copy(self, source_file: str) -> str:
        """
        Create a copy of the original file with baseline parameters.
        
        This is useful for ensuring the baseline run uses the same
        file structure as modified runs.
        
        Args:
            source_file: Path to the original SMOL file
            
        Returns:
            Path to the baseline copy
        """
        source_path = Path(source_file)
        baseline_filename = f"{source_path.stem}_baseline.smol"
        baseline_path = self.output_dir / baseline_filename
        
        # Copy file and add metadata header
        with open(source_path, 'r') as f:
            content = f.read()
        
        header = self._generate_metadata_header(source_file, {}, "Baseline configuration")
        
        with open(baseline_path, 'w') as f:
            f.write(header + content)
        
        print(f"Created baseline SMOL file: {baseline_path}")
        return str(baseline_path)


def main():
    """
    Main function for command-line usage.
    
    Example usage:
        python parameter_modifier.py simulate.smol SHALE_SIZE=25.0 TEMP_FACTOR=35.0
    """
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python parameter_modifier.py <source_file> PARAM1=VALUE1 [PARAM2=VALUE2 ...]")
        print("Example: python parameter_modifier.py simulate.smol SHALE_SIZE=25.0 TEMP_FACTOR=35.0")
        sys.exit(1)
    
    source_file = sys.argv[1]
    
    # Parse parameter changes from command line
    param_changes = {}
    for arg in sys.argv[2:]:
        if '=' in arg:
            param_name, value = arg.split('=', 1)
            try:
                # Try to parse as float, then int
                param_value = float(value)
                if param_value.is_integer():
                    param_value = int(param_value)
            except ValueError:
                print(f"Error: Invalid value for {param_name}: {value}")
                sys.exit(1)
            param_changes[param_name] = param_value
    
    # Create modifier and apply changes
    modifier = ParameterModifier()
    
    try:
        modified_file = modifier.modify_parameters(source_file, param_changes)
        print(f"Success! Modified file created: {modified_file}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()