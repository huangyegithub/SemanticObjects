#!/usr/bin/env python3
"""
Results Analyzer Module for SMOL Sensitivity Analysis

This module provides adaptive analysis capabilities that automatically adjust
based on the number of parameters being analyzed. It supports single-parameter,
few-parameter, and many-parameter analysis modes.

Author: SMOL Sensitivity Analysis Framework
Date: 2025
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from scipy import stats
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from SALib.analyze import sobol
    from SALib.sample import saltelli
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False
    print("Warning: SALib not available. Some sensitivity indices will not be computed.")


class ResultsAnalyzer:
    """
    Adaptive analyzer for SMOL simulation results.
    
    Automatically detects the number of varied parameters and applies
    appropriate analysis methods.
    """
    
    def __init__(self, output_dir: str = "../data/output"):
        """
        Initialize the results analyzer.
        
        Args:
            output_dir: Directory containing simulation output files
        """
        self.output_dir = Path(output_dir)
        self.results_df = None
        self.baseline_results = None
        self.varied_parameters = []
        self.analysis_mode = None
        self.parameter_config = self._load_parameter_config()
        
    def _load_parameter_config(self) -> Dict:
        """Load parameter configuration from JSON file."""
        config_path = Path(__file__).parent.parent / "config" / "parameters.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {"parameters": {}}
    
    def load_results(self, pattern: str = "*output*.json") -> pd.DataFrame:
        """
        Load all simulation results matching the pattern.
        
        Args:
            pattern: Glob pattern for output files
            
        Returns:
            DataFrame with consolidated results
        """
        results = []
        
        for file_path in self.output_dir.glob(pattern):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract parameters from filename
                filename = file_path.stem
                params = self._extract_parameters_from_filename(filename)
                
                # Combine with metrics
                row = {**params, **data.get('metrics', {})}
                row['filename'] = filename
                row['file_path'] = str(file_path)
                
                results.append(row)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        self.results_df = pd.DataFrame(results)
        self._detect_varied_parameters()
        self._determine_analysis_mode()
        
        return self.results_df
    
    def _extract_parameters_from_filename(self, filename: str) -> Dict:
        """Extract parameter values from filename."""
        params = {}
        
        # Check if it's a baseline file
        if 'baseline' in filename.lower():
            # Use baseline values from config
            for param_name, param_info in self.parameter_config.get('parameters', {}).items():
                params[param_name] = param_info.get('base', 0)
            params['is_baseline'] = True
        else:
            # Extract parameter values from filename
            # Format: simulate_onto_PARAM1_value1_PARAM2_value2_hash.smol
            parts = filename.split('_')
            params['is_baseline'] = False
            
            i = 0
            while i < len(parts) - 1:
                part = parts[i]
                # Check if this part is a parameter name
                if part in self.parameter_config.get('parameters', {}):
                    # Next part should be the value
                    if i + 1 < len(parts):
                        try:
                            value = float(parts[i + 1])
                            params[part] = value
                            i += 2
                        except ValueError:
                            i += 1
                else:
                    i += 1
        
        return params
    
    def _detect_varied_parameters(self):
        """Detect which parameters were varied in the analysis."""
        if self.results_df is None or self.results_df.empty:
            return
        
        # Get parameter columns (exclude metrics and metadata)
        exclude_cols = ['trap_count', 'leak_count', 'maturation_events', 
                       'migration_events', 'execution_time', 'filename', 
                       'file_path', 'is_baseline', 'final_simulation_time',
                       'status', 'warnings', 'errors']
        
        param_cols = [col for col in self.results_df.columns if col not in exclude_cols]
        
        # Find parameters that have more than one unique value
        self.varied_parameters = []
        for col in param_cols:
            if col in self.results_df.columns:
                unique_vals = self.results_df[col].dropna().unique()
                if len(unique_vals) > 1:
                    self.varied_parameters.append(col)
        
        # Identify baseline results
        baseline_mask = self.results_df.get('is_baseline', False)
        if baseline_mask.any():
            self.baseline_results = self.results_df[baseline_mask].iloc[0]
    
    def _determine_analysis_mode(self):
        """Determine the analysis mode based on number of varied parameters."""
        num_params = len(self.varied_parameters)
        
        if num_params == 0:
            self.analysis_mode = "none"
        elif num_params == 1:
            self.analysis_mode = "single"
        elif num_params <= 5:
            self.analysis_mode = "few"
        else:
            self.analysis_mode = "many"
        
        print(f"Analysis mode: {self.analysis_mode} ({num_params} varied parameters)")
        if self.varied_parameters:
            print(f"Varied parameters: {', '.join(self.varied_parameters)}")
    
    def analyze(self, target_metric: str = "trap_count") -> Dict[str, Any]:
        """
        Perform adaptive analysis based on the detected mode.
        
        Args:
            target_metric: The metric to analyze (default: trap_count)
            
        Returns:
            Dictionary containing analysis results
        """
        if self.results_df is None:
            raise ValueError("No results loaded. Call load_results() first.")
        
        results = {
            'mode': self.analysis_mode,
            'varied_parameters': self.varied_parameters,
            'target_metric': target_metric,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.analysis_mode == "none":
            results['message'] = "No parameters were varied"
            return results
        
        elif self.analysis_mode == "single":
            results.update(self._analyze_single_parameter(target_metric))
            
        elif self.analysis_mode == "few":
            results.update(self._analyze_few_parameters(target_metric))
            
        else:  # many
            results.update(self._analyze_many_parameters(target_metric))
        
        return results
    
    def _analyze_single_parameter(self, target_metric: str) -> Dict:
        """Analyze single parameter sensitivity."""
        param = self.varied_parameters[0]
        results = {'parameter': param}
        
        # Get parameter values and corresponding metrics
        df = self.results_df.dropna(subset=[param, target_metric])
        x = df[param].values
        y = df[target_metric].values
        
        if len(x) < 2:
            return {'error': 'Insufficient data points'}
        
        # Sort by parameter value
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]
        
        # Basic statistics
        results['statistics'] = {
            'min_value': float(np.min(x)),
            'max_value': float(np.max(x)),
            'min_metric': float(np.min(y)),
            'max_metric': float(np.max(y)),
            'mean_metric': float(np.mean(y)),
            'std_metric': float(np.std(y)),
            'range_effect': float(np.max(y) - np.min(y))
        }
        
        # Optimal value (where metric is maximized)
        optimal_idx = np.argmax(y)
        results['optimal'] = {
            'parameter_value': float(x[optimal_idx]),
            'metric_value': float(y[optimal_idx])
        }
        
        # Sensitivity metric
        if self.baseline_results is not None and target_metric in self.baseline_results:
            baseline_value = self.baseline_results[target_metric]
            results['sensitivity'] = {
                'absolute': float((np.max(y) - np.min(y))),
                'relative_to_baseline': float((np.max(y) - np.min(y)) / baseline_value) if baseline_value != 0 else None
            }
        
        # Correlation
        correlation, p_value = stats.pearsonr(x, y)
        results['correlation'] = {
            'pearson_r': float(correlation),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }
        
        # Monotonicity test (Spearman correlation)
        spearman_r, spearman_p = stats.spearmanr(x, y)
        results['monotonicity'] = {
            'spearman_r': float(spearman_r),
            'p_value': float(spearman_p),
            'is_monotonic': abs(spearman_r) > 0.9
        }
        
        # Fit response curve (try polynomial fits)
        results['curve_fitting'] = self._fit_response_curve(x, y)
        
        # Threshold detection
        results['thresholds'] = self._detect_thresholds(x, y)
        
        return results
    
    def _analyze_few_parameters(self, target_metric: str) -> Dict:
        """Analyze few parameters (2-5)."""
        results = {
            'parameters': self.varied_parameters,
            'pairwise_interactions': {},
            'individual_effects': {}
        }
        
        # Analyze each parameter individually
        for param in self.varied_parameters:
            df_param = self.results_df.dropna(subset=[param, target_metric])
            if len(df_param) > 1:
                x = df_param[param].values
                y = df_param[target_metric].values
                
                correlation, p_value = stats.pearsonr(x, y)
                
                results['individual_effects'][param] = {
                    'correlation': float(correlation),
                    'p_value': float(p_value),
                    'range_effect': float(np.max(y) - np.min(y)),
                    'mean_effect': float(np.mean(y))
                }
        
        # Analyze pairwise interactions
        for i, param1 in enumerate(self.varied_parameters):
            for param2 in self.varied_parameters[i+1:]:
                pair_key = f"{param1}-{param2}"
                df_pair = self.results_df.dropna(subset=[param1, param2, target_metric])
                
                if len(df_pair) > 3:
                    # Check for interaction effect using linear model
                    interaction_result = self._check_interaction(
                        df_pair[param1].values,
                        df_pair[param2].values,
                        df_pair[target_metric].values
                    )
                    results['pairwise_interactions'][pair_key] = interaction_result
        
        # Rank parameters by effect size
        effects = [(p, info['range_effect']) 
                  for p, info in results['individual_effects'].items()]
        effects.sort(key=lambda x: x[1], reverse=True)
        results['parameter_ranking'] = [p for p, _ in effects]
        
        return results
    
    def _analyze_many_parameters(self, target_metric: str) -> Dict:
        """Analyze many parameters (6+)."""
        results = {
            'parameters': self.varied_parameters,
            'summary_statistics': {},
            'parameter_importance': {},
            'correlation_matrix': {}
        }
        
        # Calculate correlation matrix
        param_data = self.results_df[self.varied_parameters + [target_metric]].dropna()
        if len(param_data) > 2:
            corr_matrix = param_data.corr()
            
            # Extract correlations with target metric
            for param in self.varied_parameters:
                if param in corr_matrix.columns:
                    results['parameter_importance'][param] = {
                        'correlation': float(corr_matrix.loc[param, target_metric]),
                        'abs_correlation': float(abs(corr_matrix.loc[param, target_metric]))
                    }
            
            # Store full correlation matrix
            results['correlation_matrix'] = corr_matrix.to_dict()
        
        # Calculate sensitivity indices if SALib is available and we have enough data
        if SALIB_AVAILABLE and len(param_data) > len(self.varied_parameters) * 2:
            try:
                si_results = self._calculate_sensitivity_indices(target_metric)
                results['sensitivity_indices'] = si_results
            except Exception as e:
                results['sensitivity_indices'] = {'error': str(e)}
        
        # Rank parameters by absolute correlation
        importance_list = [(p, info['abs_correlation']) 
                          for p, info in results['parameter_importance'].items()]
        importance_list.sort(key=lambda x: x[1], reverse=True)
        results['parameter_ranking'] = [p for p, _ in importance_list]
        
        # Summary statistics
        results['summary_statistics'] = {
            'n_simulations': len(param_data),
            'metric_mean': float(param_data[target_metric].mean()),
            'metric_std': float(param_data[target_metric].std()),
            'metric_min': float(param_data[target_metric].min()),
            'metric_max': float(param_data[target_metric].max())
        }
        
        return results
    
    def _fit_response_curve(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """Fit various response curves to the data."""
        results = {}
        
        # Linear fit
        try:
            coeffs = np.polyfit(x, y, 1)
            y_pred = np.polyval(coeffs, x)
            r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
            results['linear'] = {
                'coefficients': coeffs.tolist(),
                'r2': float(r2)
            }
        except:
            pass
        
        # Quadratic fit
        try:
            coeffs = np.polyfit(x, y, 2)
            y_pred = np.polyval(coeffs, x)
            r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
            results['quadratic'] = {
                'coefficients': coeffs.tolist(),
                'r2': float(r2)
            }
        except:
            pass
        
        # Cubic fit
        try:
            coeffs = np.polyfit(x, y, 3)
            y_pred = np.polyval(coeffs, x)
            r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
            results['cubic'] = {
                'coefficients': coeffs.tolist(),
                'r2': float(r2)
            }
        except:
            pass
        
        # Determine best fit
        best_fit = max(results.items(), key=lambda x: x[1].get('r2', 0))
        results['best_fit'] = best_fit[0]
        
        return results
    
    def _detect_thresholds(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """Detect threshold values where behavior changes significantly."""
        results = {}
        
        if len(x) < 4:
            return results
        
        # Calculate derivative (rate of change)
        dy_dx = np.gradient(y, x)
        
        # Find points with maximum rate of change
        max_change_idx = np.argmax(np.abs(dy_dx))
        results['max_change_point'] = {
            'parameter_value': float(x[max_change_idx]),
            'metric_value': float(y[max_change_idx]),
            'rate_of_change': float(dy_dx[max_change_idx])
        }
        
        # Detect plateaus (regions with minimal change)
        threshold = np.std(dy_dx) * 0.1
        plateau_mask = np.abs(dy_dx) < threshold
        
        if np.any(plateau_mask):
            plateau_regions = []
            in_plateau = False
            start_idx = 0
            
            for i, is_plateau in enumerate(plateau_mask):
                if is_plateau and not in_plateau:
                    start_idx = i
                    in_plateau = True
                elif not is_plateau and in_plateau:
                    if i - start_idx > 1:  # Minimum 2 points for a plateau
                        plateau_regions.append({
                            'start_value': float(x[start_idx]),
                            'end_value': float(x[i-1]),
                            'metric_mean': float(np.mean(y[start_idx:i]))
                        })
                    in_plateau = False
            
            if plateau_regions:
                results['plateaus'] = plateau_regions
        
        return results
    
    def _check_interaction(self, x1: np.ndarray, x2: np.ndarray, y: np.ndarray) -> Dict:
        """Check for interaction effects between two parameters."""
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        
        # Prepare data
        X = np.column_stack([x1, x2])
        
        # Fit model without interaction
        model_no_interaction = LinearRegression()
        model_no_interaction.fit(X, y)
        y_pred_no_int = model_no_interaction.predict(X)
        r2_no_interaction = model_no_interaction.score(X, y)
        
        # Fit model with interaction
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        model_with_interaction = LinearRegression()
        model_with_interaction.fit(X_poly, y)
        y_pred_with_int = model_with_interaction.predict(X_poly)
        r2_with_interaction = model_with_interaction.score(X_poly, y)
        
        # Calculate interaction strength
        interaction_strength = r2_with_interaction - r2_no_interaction
        
        return {
            'r2_without_interaction': float(r2_no_interaction),
            'r2_with_interaction': float(r2_with_interaction),
            'interaction_strength': float(interaction_strength),
            'has_significant_interaction': interaction_strength > 0.05
        }
    
    def _calculate_sensitivity_indices(self, target_metric: str) -> Dict:
        """Calculate Sobol sensitivity indices using SALib."""
        # This is a simplified version - full implementation would need
        # proper sampling design from the beginning
        results = {
            'method': 'variance_based',
            'indices': {}
        }
        
        # For each parameter, calculate a simple sensitivity index
        total_variance = self.results_df[target_metric].var()
        
        for param in self.varied_parameters:
            # Group by parameter value and calculate mean variance
            grouped = self.results_df.groupby(param)[target_metric]
            
            if len(grouped) > 1:
                # Calculate first-order index (simplified)
                means = grouped.mean()
                variance_of_means = means.var()
                
                if total_variance > 0:
                    first_order = variance_of_means / total_variance
                else:
                    first_order = 0
                
                results['indices'][param] = {
                    'first_order': float(first_order),
                    'first_order_conf': None,  # Would need bootstrap for confidence
                    'total_order': None  # Would need proper Sobol sampling
                }
        
        return results
    
    def export_results(self, analysis_results: Dict, output_file: str):
        """
        Export analysis results to JSON file.
        
        Args:
            analysis_results: Dictionary of analysis results
            output_file: Path to output file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        print(f"Analysis results saved to {output_path}")
    
    def get_summary_statistics(self, target_metric: str = "trap_count") -> pd.DataFrame:
        """
        Get summary statistics for each varied parameter.
        
        Args:
            target_metric: The metric to summarize
            
        Returns:
            DataFrame with summary statistics
        """
        if not self.varied_parameters:
            return pd.DataFrame()
        
        summary = []
        
        for param in self.varied_parameters:
            df_param = self.results_df.dropna(subset=[param, target_metric])
            
            if len(df_param) > 0:
                param_values = df_param[param].values
                metric_values = df_param[target_metric].values
                
                summary.append({
                    'parameter': param,
                    'n_samples': len(df_param),
                    'param_min': np.min(param_values),
                    'param_max': np.max(param_values),
                    'metric_mean': np.mean(metric_values),
                    'metric_std': np.std(metric_values),
                    'metric_min': np.min(metric_values),
                    'metric_max': np.max(metric_values),
                    'metric_range': np.max(metric_values) - np.min(metric_values)
                })
        
        return pd.DataFrame(summary)


def main():
    """Main function for testing the analyzer."""
    analyzer = ResultsAnalyzer()
    
    # Load results
    print("Loading simulation results...")
    df = analyzer.load_results()
    print(f"Loaded {len(df)} simulation results")
    
    # Perform analysis
    print("\nPerforming adaptive analysis...")
    results = analyzer.analyze(target_metric="trap_count")
    
    # Export results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"../data/output/analysis_results_{timestamp}.json"
    analyzer.export_results(results, output_file)
    
    # Print summary
    print("\nSummary Statistics:")
    summary = analyzer.get_summary_statistics()
    if not summary.empty:
        print(summary.to_string())
    
    return results


if __name__ == "__main__":
    results = main()