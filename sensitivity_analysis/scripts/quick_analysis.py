#!/usr/bin/env python3
"""
Quick Analysis Tool for Single Parameter Sensitivity

This script provides rapid analysis and visualization for single parameter
sensitivity analysis results. It's optimized for quick insights when analyzing
one parameter at a time.

Usage:
    python quick_analysis.py PARAMETER_NAME [--metric trap_count] [--output-dir ../data/output]

Author: SMOL Sensitivity Analysis Framework
Date: 2025
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.optimize import curve_fit
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from results_analyzer import ResultsAnalyzer


class QuickAnalyzer:
    """Quick analysis tool for single parameter sensitivity."""
    
    def __init__(self, parameter: str, metric: str = "trap_count", 
                 output_dir: str = "../data/output"):
        """
        Initialize quick analyzer.
        
        Args:
            parameter: Name of the parameter to analyze
            metric: Target metric to analyze
            output_dir: Directory containing simulation outputs
        """
        self.parameter = parameter
        self.metric = metric
        self.output_dir = Path(output_dir)
        self.figure_dir = Path("../figures/quick_analysis")
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        
    def run(self):
        """Run quick analysis for the specified parameter."""
        print(f"\n{'='*60}")
        print(f"QUICK ANALYSIS: {self.parameter}")
        print(f"{'='*60}\n")
        
        # Load results
        analyzer = ResultsAnalyzer(str(self.output_dir))
        df = analyzer.load_results()
        
        if df.empty:
            print("Error: No simulation results found!")
            return None
        
        # Check if parameter exists and was varied
        if self.parameter not in df.columns:
            print(f"Error: Parameter '{self.parameter}' not found in results!")
            print(f"Available parameters: {', '.join([col for col in df.columns if col not in ['filename', 'file_path', 'is_baseline']])}")
            return None
        
        # Filter data
        df_param = df.dropna(subset=[self.parameter, self.metric])
        unique_values = df_param[self.parameter].nunique()
        
        if unique_values < 2:
            print(f"Warning: Parameter '{self.parameter}' has only {unique_values} unique value(s).")
            print("No sensitivity analysis possible.")
            return None
        
        print(f"Found {len(df_param)} simulations with {unique_values} unique {self.parameter} values\n")
        
        # Perform analysis
        results = self._analyze(df_param)
        
        # Print results
        self._print_results(results)
        
        # Create visualization
        self._create_visualization(df_param, results)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _analyze(self, df: pd.DataFrame) -> dict:
        """Perform quick analysis on the parameter."""
        x = df[self.parameter].values
        y = df[self.metric].values
        
        # Sort by parameter value
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]
        
        results = {
            'parameter': self.parameter,
            'metric': self.metric,
            'timestamp': datetime.now().isoformat(),
            'n_simulations': len(df),
            'unique_values': len(np.unique(x))
        }
        
        # Basic statistics
        results['parameter_stats'] = {
            'min': float(np.min(x)),
            'max': float(np.max(x)),
            'mean': float(np.mean(x)),
            'std': float(np.std(x))
        }
        
        results['metric_stats'] = {
            'min': float(np.min(y)),
            'max': float(np.max(y)),
            'mean': float(np.mean(y)),
            'std': float(np.std(y)),
            'range': float(np.max(y) - np.min(y))
        }
        
        # Optimal value
        optimal_idx = np.argmax(y)
        results['optimal'] = {
            'parameter_value': float(x[optimal_idx]),
            'metric_value': float(y[optimal_idx]),
            'improvement_from_min': float(y[optimal_idx] - np.min(y)),
            'improvement_from_mean': float(y[optimal_idx] - np.mean(y))
        }
        
        # Worst value
        worst_idx = np.argmin(y)
        results['worst'] = {
            'parameter_value': float(x[worst_idx]),
            'metric_value': float(y[worst_idx])
        }
        
        # Correlation analysis
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)
        
        results['correlation'] = {
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
            'is_linear': abs(pearson_r) > 0.8,
            'is_monotonic': abs(spearman_r) > 0.8,
            'relationship': self._classify_relationship(pearson_r, spearman_r)
        }
        
        # Sensitivity metrics
        baseline_idx = len(x) // 2  # Assume middle value as baseline
        baseline_metric = y[baseline_idx]
        
        results['sensitivity'] = {
            'absolute_range': float(np.max(y) - np.min(y)),
            'relative_range': float((np.max(y) - np.min(y)) / np.mean(y)) if np.mean(y) != 0 else None,
            'coefficient_of_variation': float(np.std(y) / np.mean(y)) if np.mean(y) != 0 else None,
            'max_improvement': float(np.max(y) - baseline_metric),
            'max_decline': float(baseline_metric - np.min(y))
        }
        
        # Fit polynomial models
        results['curve_fitting'] = self._fit_curves(x, y)
        
        # Identify critical points
        results['critical_points'] = self._find_critical_points(x, y)
        
        return results
    
    def _classify_relationship(self, pearson_r: float, spearman_r: float) -> str:
        """Classify the relationship between parameter and metric."""
        if abs(pearson_r) > 0.8:
            if pearson_r > 0:
                return "strong positive linear"
            else:
                return "strong negative linear"
        elif abs(spearman_r) > 0.8:
            if spearman_r > 0:
                return "monotonic increasing (nonlinear)"
            else:
                return "monotonic decreasing (nonlinear)"
        elif abs(pearson_r) > 0.5 or abs(spearman_r) > 0.5:
            return "moderate relationship"
        elif abs(pearson_r) < 0.2 and abs(spearman_r) < 0.2:
            return "weak or no relationship"
        else:
            return "complex nonlinear"
    
    def _fit_curves(self, x: np.ndarray, y: np.ndarray) -> dict:
        """Fit various curves and find best fit."""
        results = {}
        
        # Linear fit
        try:
            coeffs = np.polyfit(x, y, 1)
            y_pred = np.polyval(coeffs, x)
            r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
            results['linear'] = {
                'coefficients': coeffs.tolist(),
                'r2': float(r2),
                'equation': f"y = {coeffs[0]:.2f}*x + {coeffs[1]:.2f}"
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
                'r2': float(r2),
                'equation': f"y = {coeffs[0]:.2e}*x² + {coeffs[1]:.2f}*x + {coeffs[2]:.2f}"
            }
        except:
            pass
        
        # Determine best fit
        if results:
            best_model = max(results.items(), key=lambda x: x[1]['r2'])
            results['best_fit'] = {
                'model': best_model[0],
                'r2': best_model[1]['r2'],
                'equation': best_model[1]['equation']
            }
        
        return results
    
    def _find_critical_points(self, x: np.ndarray, y: np.ndarray) -> dict:
        """Find critical points in the response."""
        results = {}
        
        if len(x) < 4:
            return results
        
        # Calculate gradient
        dy_dx = np.gradient(y, x)
        
        # Find steepest increase
        max_grad_idx = np.argmax(dy_dx)
        results['steepest_increase'] = {
            'parameter_value': float(x[max_grad_idx]),
            'metric_value': float(y[max_grad_idx]),
            'gradient': float(dy_dx[max_grad_idx])
        }
        
        # Find steepest decrease
        min_grad_idx = np.argmin(dy_dx)
        results['steepest_decrease'] = {
            'parameter_value': float(x[min_grad_idx]),
            'metric_value': float(y[min_grad_idx]),
            'gradient': float(dy_dx[min_grad_idx])
        }
        
        # Find plateau regions (low gradient)
        grad_threshold = np.std(dy_dx) * 0.1
        plateau_mask = np.abs(dy_dx) < grad_threshold
        
        if np.any(plateau_mask):
            plateau_indices = np.where(plateau_mask)[0]
            if len(plateau_indices) > 0:
                results['plateau_region'] = {
                    'start': float(x[plateau_indices[0]]),
                    'end': float(x[plateau_indices[-1]]),
                    'average_metric': float(np.mean(y[plateau_indices]))
                }
        
        return results
    
    def _print_results(self, results: dict):
        """Print formatted results to console."""
        print("PARAMETER STATISTICS")
        print("-" * 40)
        stats = results['parameter_stats']
        print(f"Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
        print(f"Mean ± Std: {stats['mean']:.2f} ± {stats['std']:.2f}")
        print()
        
        print("METRIC RESPONSE")
        print("-" * 40)
        m_stats = results['metric_stats']
        print(f"Range: [{m_stats['min']:.0f}, {m_stats['max']:.0f}]")
        print(f"Mean ± Std: {m_stats['mean']:.1f} ± {m_stats['std']:.1f}")
        print(f"Total variation: {m_stats['range']:.0f}")
        print()
        
        print("OPTIMAL CONFIGURATION")
        print("-" * 40)
        opt = results['optimal']
        print(f"Parameter value: {opt['parameter_value']:.2f}")
        print(f"Metric value: {opt['metric_value']:.0f}")
        print(f"Improvement from mean: {opt['improvement_from_mean']:.0f}")
        print(f"Improvement from min: {opt['improvement_from_min']:.0f}")
        print()
        
        print("SENSITIVITY ANALYSIS")
        print("-" * 40)
        sens = results['sensitivity']
        print(f"Absolute sensitivity: {sens['absolute_range']:.0f}")
        if sens['relative_range']:
            print(f"Relative sensitivity: {sens['relative_range']*100:.1f}%")
        if sens['coefficient_of_variation']:
            print(f"Coefficient of variation: {sens['coefficient_of_variation']:.3f}")
        print()
        
        print("CORRELATION ANALYSIS")
        print("-" * 40)
        corr = results['correlation']
        print(f"Relationship type: {corr['relationship']}")
        print(f"Pearson correlation: {corr['pearson_r']:.3f} (p={corr['pearson_p']:.3e})")
        print(f"Spearman correlation: {corr['spearman_r']:.3f} (p={corr['spearman_p']:.3e})")
        print()
        
        if 'curve_fitting' in results and 'best_fit' in results['curve_fitting']:
            print("BEST FIT MODEL")
            print("-" * 40)
            best = results['curve_fitting']['best_fit']
            print(f"Model: {best['model']}")
            print(f"R² score: {best['r2']:.4f}")
            print(f"Equation: {best['equation']}")
            print()
    
    def _create_visualization(self, df: pd.DataFrame, results: dict):
        """Create comprehensive visualization."""
        x = df[self.parameter].values
        y = df[self.metric].values
        
        # Sort for plotting
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Main response curve with fit
        ax = axes[0, 0]
        ax.scatter(x, y, alpha=0.6, s=50, label='Simulations')
        
        # Add best fit curve
        if 'curve_fitting' in results and 'best_fit' in results['curve_fitting']:
            x_smooth = np.linspace(x.min(), x.max(), 200)
            best_model = results['curve_fitting']['best_fit']['model']
            coeffs = results['curve_fitting'][best_model]['coefficients']
            y_smooth = np.polyval(coeffs, x_smooth)
            ax.plot(x_smooth, y_smooth, 'r-', linewidth=2, 
                   label=f"{best_model} fit (R²={results['curve_fitting']['best_fit']['r2']:.3f})")
        
        # Mark optimal point
        opt = results['optimal']
        ax.plot(opt['parameter_value'], opt['metric_value'], 'g*', 
               markersize=15, label=f"Optimal ({opt['parameter_value']:.2f})")
        
        # Mark worst point
        worst = results['worst']
        ax.plot(worst['parameter_value'], worst['metric_value'], 'rx', 
               markersize=12, label=f"Worst ({worst['parameter_value']:.2f})")
        
        ax.set_xlabel(self.parameter, fontsize=12)
        ax.set_ylabel(self.metric, fontsize=12)
        ax.set_title('Response Curve', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Gradient (rate of change)
        ax = axes[0, 1]
        if len(x) > 3:
            dy_dx = np.gradient(y, x)
            ax.plot(x, dy_dx, 'o-', linewidth=2, markersize=6)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            # Mark critical points
            if 'critical_points' in results and 'steepest_increase' in results['critical_points']:
                steep = results['critical_points']['steepest_increase']
                ax.plot(steep['parameter_value'], steep['gradient'], 'g^', 
                       markersize=10, label='Max increase')
            
            ax.set_xlabel(self.parameter, fontsize=12)
            ax.set_ylabel(f'd({self.metric})/d({self.parameter})', fontsize=12)
            ax.set_title('Rate of Change', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. Box plot by parameter bins
        ax = axes[1, 0]
        n_bins = min(5, len(np.unique(x)))
        if n_bins > 1:
            df_copy = df.copy()
            df_copy['param_bin'] = pd.qcut(df_copy[self.parameter], n_bins, duplicates='drop')
            df_copy.boxplot(column=self.metric, by='param_bin', ax=ax)
            ax.set_xlabel(f'{self.parameter} (binned)', fontsize=12)
            ax.set_ylabel(self.metric, fontsize=12)
            ax.set_title('Distribution by Parameter Range', fontsize=14, fontweight='bold')
            plt.sca(ax)
            plt.xticks(rotation=45)
        
        # 4. Summary text
        ax = axes[1, 1]
        ax.axis('off')
        
        summary = f"QUICK ANALYSIS SUMMARY\n{'='*30}\n\n"
        summary += f"Parameter: {self.parameter}\n"
        summary += f"Metric: {self.metric}\n"
        summary += f"Simulations: {results['n_simulations']}\n\n"
        
        summary += f"OPTIMAL POINT\n"
        summary += f"  Value: {opt['parameter_value']:.2f}\n"
        summary += f"  Result: {opt['metric_value']:.0f}\n\n"
        
        summary += f"SENSITIVITY\n"
        sens = results['sensitivity']
        summary += f"  Range: {sens['absolute_range']:.0f}\n"
        if sens['relative_range']:
            summary += f"  Relative: {sens['relative_range']*100:.1f}%\n\n"
        
        summary += f"RELATIONSHIP\n"
        summary += f"  Type: {results['correlation']['relationship']}\n"
        
        if 'curve_fitting' in results and 'best_fit' in results['curve_fitting']:
            summary += f"\nBEST FIT\n"
            best = results['curve_fitting']['best_fit']
            summary += f"  Model: {best['model']}\n"
            summary += f"  R²: {best['r2']:.4f}\n"
        
        ax.text(0.1, 0.9, summary, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle(f'Quick Analysis: {self.parameter}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.figure_dir / f"{self.parameter}_{timestamp}.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualization saved to: {filename}")
    
    def _save_results(self, results: dict):
        """Save analysis results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"quick_analysis_{self.parameter}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {filename}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Quick sensitivity analysis for single parameter'
    )
    parser.add_argument('parameter', help='Name of parameter to analyze')
    parser.add_argument('--metric', default='trap_count', 
                       help='Target metric (default: trap_count)')
    parser.add_argument('--output-dir', default='../data/output',
                       help='Directory containing simulation outputs')
    
    args = parser.parse_args()
    
    # Run quick analysis
    analyzer = QuickAnalyzer(args.parameter, args.metric, args.output_dir)
    results = analyzer.run()
    
    if results:
        print(f"\n{'='*60}")
        print("Quick analysis complete!")
        print(f"{'='*60}")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())