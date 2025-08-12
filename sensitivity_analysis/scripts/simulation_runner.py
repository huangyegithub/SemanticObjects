#!/usr/bin/env python3
"""
Simulation Runner Module for SMOL Sensitivity Analysis

This module executes SMOL simulations and extracts key metrics from the output.
It handles running the SMOL JAR file, capturing output, and parsing results
for trap/leak counts and other relevant metrics.

Author: Sensitivity Analysis Tool
Date: 2024
"""

import subprocess
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import time
import os
import sys


class SimulationRunner:
    """
    Executes SMOL simulations and extracts metrics from output.
    
    This class handles:
    - Running SMOL simulations via the Java JAR
    - Capturing and parsing simulation output
    - Extracting key metrics (traps, leaks, warnings)
    - Saving results in structured format
    """
    
    def __init__(self, 
                 jar_path: str = None,
                 output_dir: str = "../data/output",
                 timeout: int = 1800,  # 30 minutes for real simulations
                 verbose: bool = True):
        """
        Initialize the simulation runner.
        
        Args:
            jar_path: Path to the SMOL JAR file (auto-detected if None)
            output_dir: Directory for simulation output files
            timeout: Maximum simulation time in seconds (default: 30 minutes)
            verbose: Enable verbose debugging output
        """
        self.verbose = verbose
        
        if self.verbose:
            print(f"[DEBUG] Initializing SimulationRunner...")
            print(f"[DEBUG] Output directory: {output_dir}")
            print(f"[DEBUG] Timeout: {timeout} seconds")
            print(f"[DEBUG] Verbose mode: {verbose}")
        
        self.jar_path = self._find_jar_path(jar_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        
        if self.verbose:
            print(f"[DEBUG] SimulationRunner initialized successfully")
            print(f"[DEBUG] JAR path: {self.jar_path}")
            print(f"[DEBUG] Output directory created: {self.output_dir}")
            print(f"[DEBUG] " + "="*50)
        
    def _find_jar_path(self, jar_path: Optional[str]) -> str:
        """
        Find the SMOL JAR file path.
        
        Args:
            jar_path: Explicitly provided JAR path or None
            
        Returns:
            Path to the SMOL JAR file
            
        Raises:
            FileNotFoundError: If JAR file cannot be found
        """
        if self.verbose:
            print(f"[DEBUG] Looking for SMOL JAR...")
        
        if jar_path:
            if self.verbose:
                print(f"[DEBUG] Checking explicitly provided path: {jar_path}")
            if Path(jar_path).exists():
                if self.verbose:
                    print(f"[DEBUG] ✓ Found JAR at provided path: {jar_path}")
                return jar_path
            else:
                if self.verbose:
                    print(f"[DEBUG] ✗ Provided path does not exist: {jar_path}")
        
        # Try common locations relative to project root
        possible_paths = [
            "../../build/libs/smol.jar",
            "../../../build/libs/smol.jar",
            "../../../../build/libs/smol.jar",
            "/Users/yehuang/Desktop/CEED/Starting_from_June/SemanticObjects/build/libs/smol.jar"
        ]
        
        if self.verbose:
            print(f"[DEBUG] Searching in {len(possible_paths)} common locations...")
        
        for i, path in enumerate(possible_paths, 1):
            full_path = Path(path).resolve()
            if self.verbose:
                print(f"[DEBUG] [{i}/{len(possible_paths)}] Checking: {full_path}")
            
            if full_path.exists():
                if self.verbose:
                    print(f"[DEBUG] ✓ Found SMOL JAR at: {full_path}")
                else:
                    print(f"Found SMOL JAR at: {full_path}")
                return str(full_path)
            else:
                if self.verbose:
                    print(f"[DEBUG] ✗ Not found: {full_path}")
        
        error_msg = (
            "Could not find smol.jar. Please provide the path explicitly or "
            "ensure the project is built (./gradlew build)"
        )
        if self.verbose:
            print(f"[DEBUG] ERROR: {error_msg}")
        raise FileNotFoundError(error_msg)
    
    def run_simulation(self, 
                      smol_file: str,
                      save_output: bool = True,
                      verbose: bool = True) -> Dict[str, Any]:
        """
        Run a SMOL simulation and extract metrics.
        
        Args:
            smol_file: Path to the SMOL file to execute
            save_output: Whether to save raw output to file
            verbose: Whether to print progress information
            
        Returns:
            Dictionary containing:
                - trap_count: Number of hydrocarbon traps
                - leak_count: Number of hydrocarbon leaks
                - execution_time: Simulation runtime in seconds
                - warnings: List of warning messages
                - errors: List of error messages
                - raw_output: Full simulation output (if requested)
                - parameters: Parameter values used (if available)
                
        Example:
            >>> runner = SimulationRunner()
            >>> results = runner.run_simulation("modified_sim.smol")
            >>> print(f"Traps: {results['trap_count']}, Leaks: {results['leak_count']}")
        """
        smol_path = Path(smol_file)
        
        if self.verbose or verbose:
            print(f"[DEBUG] " + "="*50)
            print(f"[DEBUG] Starting simulation run...")
            print(f"[DEBUG] SMOL file: {smol_file}")
            print(f"[DEBUG] Resolved path: {smol_path.resolve()}")
            print(f"[DEBUG] Save output: {save_output}")
            print(f"[DEBUG] Verbose: {verbose}")
        
        if not smol_path.exists():
            if self.verbose or verbose:
                print(f"[DEBUG] ERROR: SMOL file not found: {smol_file}")
            raise FileNotFoundError(f"SMOL file not found: {smol_file}")
        
        if self.verbose or verbose:
            print(f"[DEBUG] ✓ SMOL file exists and is readable")
            print(f"[DEBUG] File size: {smol_path.stat().st_size} bytes")
        
        if verbose:
            print(f"Running simulation: {smol_path.name}")
        
        # Prepare command with all required flags matching execute_case.sh
        cmd = [
            "java", "-jar", self.jar_path,
            "-i", str(smol_path.resolve()),
            "-v",
            "-e", 
            "-b", "examples/Geological/total_mini.ttl",
            "-p", "UFRGS1=https://www.inf.ufrgs.br/bdi/ontologies/geocoreontology#UFRGS",
            "-p", "obo=http://purl.obolibrary.org/obo/",
            "-d", "http://www.semanticweb.org/quy/ontologies/2023/2/untitled-ontology-38#"
        ]

        if self.verbose or verbose:
            print(f"[DEBUG] Command to execute: {' '.join(cmd)}")
            print(f"[DEBUG] Working directory: {smol_path.parent}")
            print(f"[DEBUG] Timeout: {self.timeout} seconds ({self.timeout/60:.1f} minutes)")
            print(f"[DEBUG] Starting subprocess...")
        
        # Run simulation
        start_time = time.time()
        try:
            if self.verbose or verbose:
                print(f"[DEBUG] Subprocess starting at {time.strftime('%H:%M:%S')}")
                print(f"[DEBUG] Running geological simulation...")
                print(f"[DEBUG] This may take several minutes - please wait...")
            elif verbose:
                print(f"Running geological simulation (timeout: {self.timeout/60:.1f} minutes)...")
            
            # Let simulation run completely, then exit
            # The main function should execute automatically when the file is loaded

            # Use Popen for real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(Path(self.jar_path).parent.parent.parent),
                bufsize=1  # Line buffered
            )
            # Collect output while printing in real-time
            output_lines = []
            error_lines = []

            # Read stdout line by line
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    print(line.rstrip())  # Print to terminal (remove extra newline)
                    output_lines.append(line)

            # Get any remaining stderr
            stderr_output = process.stderr.read()
            if stderr_output:
                error_lines.append(stderr_output)
                if self.verbose:
                    print(f"[STDERR] {stderr_output}")

            # Wait for process to complete
            process.wait()
            execution_time = time.time() - start_time

            # Combine output for compatibility with rest of code
            output = ''.join(output_lines)
            error_output = ''.join(error_lines)

            # Create result object to match subprocess.run interface
            class Result:
                def __init__(self, stdout, stderr, returncode):
                    self.stdout = stdout
                    self.stderr = stderr
                    self.returncode = returncode

            result = Result(output, error_output, process.returncode)
            
            if self.verbose or verbose:
                print(f"[DEBUG] Subprocess completed at {time.strftime('%H:%M:%S')}")
                print(f"[DEBUG] Execution time: {execution_time:.3f} seconds")
                print(f"[DEBUG] Return code: {result.returncode}")
                print(f"[DEBUG] Stdout length: {len(result.stdout)} characters")
                print(f"[DEBUG] Stderr length: {len(result.stderr)} characters")
            
        except subprocess.TimeoutExpired:
            if self.verbose or verbose:
                print(f"[DEBUG] ⚠️  TIMEOUT: Process exceeded {self.timeout} seconds")
                print(f"[DEBUG] Returning timeout result...")
            
            return {
                "trap_count": 0,
                "leak_count": 0,
                "execution_time": self.timeout,
                "warnings": [],
                "errors": [f"Simulation timeout after {self.timeout} seconds"],
                "status": "timeout"
            }
        
        # Parse output
        output = result.stdout
        error_output = result.stderr
        
        if self.verbose or verbose:
            print(f"[DEBUG] " + "-"*30)
            print(f"[DEBUG] Parsing output...")
            print(f"[DEBUG] Raw stdout (first 200 chars):")
            print(f"[DEBUG] {repr(output[:200])}")
            print(f"[DEBUG] Raw stderr (first 200 chars):")
            print(f"[DEBUG] {repr(error_output[:200])}")
            print(f"[DEBUG] " + "-"*30)
        
        # Extract metrics
        if self.verbose or verbose:
            print(f"[DEBUG] Extracting metrics...")
        
        metrics = self._extract_metrics(output, error_output)
        metrics["execution_time"] = execution_time
        metrics["smol_file"] = str(smol_file)
        metrics["timestamp"] = datetime.now().isoformat()
        metrics["return_code"] = result.returncode
        
        if self.verbose or verbose:
            print(f"[DEBUG] Extracted metrics:")
            print(f"[DEBUG]   - Status: {metrics.get('status', 'unknown')}")
            print(f"[DEBUG]   - Trap count: {metrics.get('trap_count', 0)}")
            print(f"[DEBUG]   - Leak count: {metrics.get('leak_count', 0)}")
            print(f"[DEBUG]   - Errors: {len(metrics.get('errors', []))}")
            print(f"[DEBUG]   - Warnings: {len(metrics.get('warnings', []))}")
        
        # Extract parameters from filename or metadata
        if self.verbose or verbose:
            print(f"[DEBUG] Extracting parameters from file...")
        
        metrics["parameters"] = self._extract_parameters(smol_path)
        
        if self.verbose or verbose:
            print(f"[DEBUG] Found {len(metrics['parameters'])} parameters")
        
        # Save output if requested
        if save_output:
            if self.verbose or verbose:
                print(f"[DEBUG] Saving output to file...")
            
            output_file = self._save_output(smol_path, output, error_output, metrics)
            metrics["output_file"] = str(output_file)
            
            if self.verbose or verbose:
                print(f"[DEBUG] Output saved to: {output_file}")
        
        if verbose:
            self._print_summary(metrics)
        
        if self.verbose or verbose:
            print(f"[DEBUG] Simulation run completed successfully!")
            print(f"[DEBUG] " + "="*50)
        
        return metrics
    
    def _extract_metrics(self, output: str, error_output: str) -> Dict[str, Any]:
        """
        Extract key metrics from simulation output.
        
        Args:
            output: Standard output from simulation
            error_output: Standard error from simulation
            
        Returns:
            Dictionary of extracted metrics
        """
        metrics = {
            "trap_count": 0,
            "leak_count": 0,
            "warnings": [],
            "errors": [],
            "maturation_events": 0,
            "migration_events": 0,
            "status": "completed"
        }
        
        # Count traps and leaks
        trap_pattern = r'\btrap\b'
        leak_pattern = r'\bleak\b'
        
        metrics["trap_count"] = len(re.findall(trap_pattern, output, re.IGNORECASE))
        metrics["leak_count"] = len(re.findall(leak_pattern, output, re.IGNORECASE))
        
        # Count maturation events
        maturation_pattern = r'maturation on-going!'
        metrics["maturation_events"] = len(re.findall(maturation_pattern, output))
        
        # Count migration events
        migration_pattern = r'migrate from shale'
        metrics["migration_events"] = len(re.findall(migration_pattern, output))
        
        # Extract warnings (lines containing "warning" or "warn")
        warning_lines = [
            line.strip() 
            for line in output.split('\n') 
            if 'warning' in line.lower() or 'warn' in line.lower()
        ]
        metrics["warnings"] = warning_lines[:10]  # Limit to first 10
        
        # Extract errors from stderr
        if error_output:
            error_lines = [
                line.strip() 
                for line in error_output.split('\n') 
                if line.strip()
            ]
            metrics["errors"] = error_lines[:10]  # Limit to first 10
            if error_lines:
                metrics["status"] = "error"
        
        # Extract final simulation time
        time_pattern = r'Ending simulation with t =[\s-]*([\d.]+)'
        time_match = re.search(time_pattern, output)
        if time_match:
            metrics["final_simulation_time"] = float(time_match.group(1))
        
        # Extract evaluation completion message
        eval_pattern = r'Evaluation took ([\d.]+) seconds'
        eval_match = re.search(eval_pattern, output)
        if eval_match:
            metrics["evaluation_time"] = float(eval_match.group(1))
            # If we see evaluation completion, simulation likely ran successfully
            if metrics["status"] == "error" and not metrics["errors"]:
                metrics["status"] = "completed"
        
        return metrics
    
    def _extract_parameters(self, smol_path: Path) -> Dict[str, Any]:
        """
        Extract parameter information from SMOL file.
        
        Args:
            smol_path: Path to the SMOL file
            
        Returns:
            Dictionary of parameter values
        """
        parameters = {}
        
        # Try to extract from metadata header
        try:
            with open(smol_path, 'r') as f:
                content = f.read()
            
            # Look for modified parameters in header
            param_pattern = r'// - (\w+): ([\d.-]+) -> ([\d.-]+)'
            matches = re.findall(param_pattern, content)
            
            for param_name, baseline, modified in matches:
                parameters[param_name] = {
                    "baseline": float(baseline),
                    "modified": float(modified)
                }
        except Exception:
            pass  # If extraction fails, return empty dict
        
        return parameters
    
    def _save_output(self, 
                    smol_path: Path, 
                    output: str, 
                    error_output: str,
                    metrics: Dict[str, Any]) -> Path:
        """
        Save simulation output to file.
        
        Args:
            smol_path: Path to the SMOL file
            output: Standard output
            error_output: Standard error
            metrics: Extracted metrics
            
        Returns:
            Path to the saved output file
        """
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{smol_path.stem}_output_{timestamp}.json"
        output_path = self.output_dir / output_filename
        
        # Prepare output data
        output_data = {
            "simulation": {
                "smol_file": str(smol_path),
                "timestamp": metrics["timestamp"],
                "execution_time": metrics["execution_time"]
            },
            "metrics": metrics,
            "raw_output": {
                "stdout": output,
                "stderr": error_output
            }
        }
        
        # Save to JSON file
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        return output_path
    
    def _print_summary(self, metrics: Dict[str, Any]) -> None:
        """Print a summary of simulation results."""
        print(f"\nSimulation Results:")
        print(f"  Status: {metrics['status']}")
        print(f"  Execution time: {metrics['execution_time']:.2f} seconds")
        print(f"  Traps: {metrics['trap_count']}")
        print(f"  Leaks: {metrics['leak_count']}")
        print(f"  Maturation events: {metrics['maturation_events']}")
        print(f"  Migration events: {metrics['migration_events']}")
        
        if metrics['warnings']:
            print(f"  Warnings: {len(metrics['warnings'])}")
        
        if metrics['errors']:
            print(f"  Errors: {len(metrics['errors'])}")
    
    def run_baseline(self, baseline_smol: str) -> Dict[str, Any]:
        """
        Run baseline simulation for comparison.
        
        Args:
            baseline_smol: Path to baseline SMOL file
            
        Returns:
            Baseline simulation metrics
        """
        print("\n" + "="*50)
        print("Running BASELINE simulation")
        print("="*50)
        
        return self.run_simulation(baseline_smol, verbose=True)


def main():
    """
    Main function for command-line usage.
    
    Example usage:
        python simulation_runner.py path/to/simulation.smol
        python simulation_runner.py path/to/simulation.smol --verbose
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SMOL simulation and extract metrics")
    parser.add_argument("smol_file", help="Path to the SMOL file to execute")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debugging output")
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout in seconds (default: 1800)")
    
    args = parser.parse_args()
    
    # Create runner and execute simulation
    runner = SimulationRunner(verbose=args.verbose, timeout=args.timeout)
    
    try:
        results = runner.run_simulation(args.smol_file, verbose=args.verbose)
        
        # Print summary
        if not args.verbose:
            print("\nSimulation completed successfully!")
        print(f"Results saved to: {results.get('output_file', 'Not saved')}")
        
    except Exception as e:
        if args.verbose:
            print(f"[DEBUG] EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
        else:
            print(f"Error running simulation: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()