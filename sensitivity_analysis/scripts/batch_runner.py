#!/usr/bin/env python3
"""
Batch Runner Module for SMOL Sensitivity Analysis

This module orchestrates running multiple SMOL simulations with different
parameter values. It implements one-at-a-time (OAT) sensitivity analysis
by varying each parameter while keeping others at baseline.

Author: Sensitivity Analysis Tool
Date: 2024
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import multiprocessing

# === ORIGINAL IMPORTS ===
# from parameter_modifier import ParameterModifier

# === ENHANCED IMPORTS: Support both original and template-based modifier ===
try:
    from parameter_modifier_template import TemplateParameterModifier
    USE_TEMPLATE_MODIFIER = True
except ImportError:
    from parameter_modifier import ParameterModifier
    USE_TEMPLATE_MODIFIER = False
    
from simulation_runner import SimulationRunner


class BatchRunner:
    """
    Orchestrates batch simulation runs for sensitivity analysis.
    
    This class handles:
    - Generating parameter variations for OAT analysis
    - Running simulations in parallel
    - Collecting and organizing results
    - Saving results in structured format
    """
    
    def __init__(self,
                 source_smol: str,
                 config_dir: str = "../config",
                 output_dir: str = "../data/output",
                 max_workers: Optional[int] = None):
        """
        Initialize the batch runner.
        
        Args:
            source_smol: Path to the original SMOL file
            config_dir: Directory containing parameter configurations
            output_dir: Directory for processed results
            max_workers: Maximum parallel workers (default: CPU count - 1)
        """
        self.source_smol = Path(source_smol)
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up max workers
        if max_workers is None:
            self.max_workers = max(1, multiprocessing.cpu_count() - 1)
        else:
            self.max_workers = max_workers
        
        # Initialize components
        # === ENHANCED: Use template modifier if available ===
        if USE_TEMPLATE_MODIFIER:
            self.modifier = TemplateParameterModifier(config_dir)
            # Check if source file is a template
            self.is_template = "template" in source_smol or "{{" in Path(source_smol).read_text()
        else:
            # === ORIGINAL CODE ===
            self.modifier = ParameterModifier(config_dir)
            self.is_template = False
        self.runner = SimulationRunner()
        
        # Load configuration
        self._load_config()
        
    def _load_config(self) -> None:
        """Load parameter configuration."""
        with open(self.config_dir / "parameters.json", 'r') as f:
            self.params_config = json.load(f)
        
        with open(self.config_dir / "baseline.json", 'r') as f:
            self.baseline_config = json.load(f)
    
    def run_oat_analysis(self, 
                        n_samples: int = 5,
                        parameters: Optional[List[str]] = None,
                        parallel: bool = True) -> pd.DataFrame:
        """
        Run One-at-a-Time (OAT) sensitivity analysis.
        
        Args:
            n_samples: Number of samples per parameter
            parameters: List of parameters to analyze (None = all)
            parallel: Whether to run simulations in parallel
            
        Returns:
            DataFrame with simulation results
            
        Example:
            >>> batch = BatchRunner("simulate.smol")
            >>> results = batch.run_oat_analysis(n_samples=5)
            >>> print(results.groupby('parameter')['trap_count'].mean())
        """
        print(f"\n{'='*60}")
        print(f"Starting OAT Sensitivity Analysis")
        print(f"{'='*60}")
        print(f"Source file: {self.source_smol}")
        print(f"Parameters: {len(self.params_config['parameters'])}")
        print(f"Samples per parameter: {n_samples}")
        print(f"Max parallel workers: {self.max_workers if parallel else 1}")
        print(f"{'='*60}\n")
        
        # Get parameters to analyze
        if parameters is None:
            parameters = list(self.params_config['parameters'].keys())
        
        # Generate parameter variations
        variations = self._generate_oat_variations(parameters, n_samples)
        
        # Add baseline run
        baseline_variation = {
            'parameter': 'baseline',
            'value': 0,
            'param_changes': {},
            'description': 'Baseline configuration'
        }
        variations.insert(0, baseline_variation)
        
        print(f"Total simulations to run: {len(variations)}")
        
        # Run simulations
        if parallel and self.max_workers > 1:
            results = self._run_parallel(variations)
        else:
            results = self._run_sequential(variations)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        self._save_results(results_df)
        
        # Print summary
        self._print_summary(results_df)
        
        return results_df
    
    def _generate_oat_variations(self, 
                                parameters: List[str], 
                                n_samples: int) -> List[Dict[str, Any]]:
        """
        Generate parameter variations for OAT analysis.
        
        Args:
            parameters: List of parameters to vary
            n_samples: Number of samples per parameter
            
        Returns:
            List of variation dictionaries
        """
        variations = []
        
        for param_name in parameters:
            param_info = self.params_config['parameters'][param_name]
            min_val = param_info['min']
            max_val = param_info['max']
            baseline_val = param_info['base']
            
            # Generate sample values
            if param_info['type'] == 'int':
                # For integers, ensure we get distinct values
                sample_values = np.linspace(min_val, max_val, n_samples)
                sample_values = np.unique(np.round(sample_values).astype(int))
            else:
                sample_values = np.linspace(min_val, max_val, n_samples)
            
            # Create variations
            for value in sample_values:
                # Skip if too close to baseline (within 1%)
                if abs(value - baseline_val) / baseline_val < 0.01:
                    continue
                
                variation = {
                    'parameter': param_name,
                    'value': float(value) if param_info['type'] == 'float' else int(value),
                    'param_changes': {param_name: value},
                    'description': f"{param_name} = {value}"
                }
                variations.append(variation)
        
        return variations
    
    def _run_sequential(self, variations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run simulations sequentially.
        
        Args:
            variations: List of parameter variations
            
        Returns:
            List of simulation results
        """
        results = []
        
        with tqdm(total=len(variations), desc="Running simulations") as pbar:
            for variation in variations:
                result = self._run_single_variation(variation)
                results.append(result)
                pbar.update(1)
        
        return results
    
    def _run_parallel(self, variations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run simulations in parallel.
        
        Args:
            variations: List of parameter variations
            
        Returns:
            List of simulation results
        """
        results = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_variation = {
                executor.submit(self._run_single_variation, variation): variation
                for variation in variations
            }
            
            # Process completed tasks
            with tqdm(total=len(variations), desc="Running simulations") as pbar:
                for future in concurrent.futures.as_completed(future_to_variation):
                    try:
                        result = future.result()
                        results.append(result)
                        pbar.update(1)
                    except Exception as e:
                        variation = future_to_variation[future]
                        print(f"\nError in simulation {variation['description']}: {e}")
                        # Add failed result
                        failed_result = {
                            'parameter': variation['parameter'],
                            'value': variation['value'],
                            'status': 'failed',
                            'error': str(e),
                            'trap_count': 0,
                            'leak_count': 0
                        }
                        results.append(failed_result)
                        pbar.update(1)
        
        return results
    
    def _run_single_variation(self, variation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single parameter variation.
        
        Args:
            variation: Parameter variation dictionary
            
        Returns:
            Simulation results with parameter information
        """
        # Create modified SMOL file (or use baseline)
        if variation['param_changes']:
            # === ENHANCED: Use appropriate modifier based on file type ===
            if USE_TEMPLATE_MODIFIER and self.is_template:
                # Use template-based modification
                modifier = TemplateParameterModifier(str(self.config_dir))
                modified_file = modifier.modify_parameters_from_template(
                    str(self.source_smol),
                    variation['param_changes'],
                    variation['description']
                )
            else:
                # === ORIGINAL CODE: Use regex-based modification ===
                modifier = ParameterModifier(str(self.config_dir))
                modified_file = modifier.modify_parameters(
                    str(self.source_smol),
                    variation['param_changes'],
                    variation['description']
                )
        else:
            # Baseline run
            if USE_TEMPLATE_MODIFIER and self.is_template:
                # Create baseline from template
                modifier = TemplateParameterModifier(str(self.config_dir))
                modified_file = modifier.create_baseline_from_template(str(self.source_smol))
            else:
                # === ORIGINAL CODE ===
                modifier = ParameterModifier(str(self.config_dir))
                modified_file = modifier.create_baseline_copy(str(self.source_smol))
        
        # Run simulation
        # Need to recreate runner for multiprocessing
        runner = SimulationRunner()
        sim_results = runner.run_simulation(modified_file, save_output=True, verbose=True)
        
        # Combine results
        result = {
            'parameter': variation['parameter'],
            'value': variation.get('value', 'baseline'),
            **sim_results
        }
        
        return result
    
    def _save_results(self, results_df: pd.DataFrame) -> None:
        """
        Save results to CSV, JSON formats, and create debug log.
        
        Args:
            results_df: DataFrame with simulation results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as CSV
        csv_path = self.output_dir / f"sensitivity_results_{timestamp}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        # Save as JSON with metadata
        json_data = {
            'metadata': {
                'source_file': str(self.source_smol),
                'timestamp': timestamp,
                'total_simulations': len(results_df),
                'parameters_analyzed': results_df['parameter'].nunique() - 1  # Exclude baseline
            },
            'results': results_df.to_dict('records')
        }
        
        json_path = self.output_dir / f"sensitivity_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Results saved to: {json_path}")
        
        # Save debug log with all output
        debug_path = self.output_dir / f"debug_log_{timestamp}.txt"
        with open(debug_path, 'w') as f:
            f.write(f"SMOL SENSITIVITY ANALYSIS DEBUG LOG\n")
            f.write(f"{'='*60}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Source file: {self.source_smol}\n")
            f.write(f"Parameters analysed: {results_df['parameter'].nunique() - 1}\n")
            f.write(f"Total simulations: {len(results_df)}\n")
            f.write(f"{'='*60}\n\n")
            
            # Write detailed results for each simulation
            for _, row in results_df.iterrows():
                f.write(f"SIMULATION: {row['parameter']} = {row['value']}\n")
                f.write(f"Status: {row.get('status', 'unknown')}\n")
                f.write(f"Trap Count: {row.get('trap_count', 'N/A')}\n")
                f.write(f"Leak Count: {row.get('leak_count', 'N/A')}\n")
                f.write(f"Execution Time: {row.get('execution_time', 'N/A')}s\n")
                if 'error' in row:
                    f.write(f"ERROR: {row['error']}\n")
                f.write("-" * 40 + "\n")
            
            f.write("\nCSV SUMMARY:\n")
            f.write(results_df.to_string())
        
        print(f"Debug log saved to: {debug_path}")
    
    def _print_summary(self, results_df: pd.DataFrame) -> None:
        """Print summary of batch run results."""
        print(f"\n{'='*60}")
        print("BATCH RUN SUMMARY")
        print(f"{'='*60}")
        
        # Overall statistics
        print(f"\nTotal simulations: {len(results_df)}")
        print(f"Successful: {len(results_df[results_df['status'] == 'completed'])}")
        print(f"Failed: {len(results_df[results_df['status'] != 'completed'])}")
        
        # Baseline results
        baseline = results_df[results_df['parameter'] == 'baseline'].iloc[0]
        print(f"\nBaseline results:")
        print(f"  Traps: {baseline['trap_count']}")
        print(f"  Leaks: {baseline['leak_count']}")
        
        # Parameter impact summary
        print(f"\nParameter impact on trap count:")
        for param in results_df['parameter'].unique():
            if param == 'baseline':
                continue
            
            param_results = results_df[results_df['parameter'] == param]
            trap_range = param_results['trap_count'].max() - param_results['trap_count'].min()
            print(f"  {param}: range = {trap_range}")
        
        print(f"{'='*60}\n")


def main():
    """
    Main function for command-line usage.
    
    Example usage:
        python batch_runner.py path/to/simulate.smol --samples 5 --parallel
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run batch SMOL simulations for sensitivity analysis"
    )
    parser.add_argument(
        "smol_file",
        help="Path to the source SMOL file"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of samples per parameter (default: 5)"
    )
    parser.add_argument(
        "--parameters",
        nargs="+",
        help="Specific parameters to analyze (default: all)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run simulations in parallel"
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Maximum number of parallel workers"
    )
    
    args = parser.parse_args()
    
    # Create batch runner
    batch_runner = BatchRunner(
        args.smol_file,
        max_workers=args.workers
    )
    
    # Run analysis
    try:
        results = batch_runner.run_oat_analysis(
            n_samples=args.samples,
            parameters=args.parameters,
            parallel=args.parallel
        )
        
        print("\nBatch run completed successfully!")
        
    except Exception as e:
        print(f"Error in batch run: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()