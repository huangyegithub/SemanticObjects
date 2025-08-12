#!/usr/bin/env python3
"""
Sensitivity Plots Module for SMOL Analysis

This module provides adaptive visualization capabilities that automatically
adjust based on the number of parameters being analyzed. It creates different
plot types for single, few, and many parameter analyses.

Author: SMOL Sensitivity Analysis Framework
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class SensitivityPlotter:
    """
    Adaptive plotter for sensitivity analysis results.
    
    Automatically creates appropriate visualizations based on the
    analysis mode (single, few, or many parameters).
    """
    
    def __init__(self, output_dir: str = "../data/output", figure_dir: str = "../figures"):
        """
        Initialize the plotter.
        
        Args:
            output_dir: Directory containing analysis results
            figure_dir: Directory to save figures
        """
        self.output_dir = Path(output_dir)
        self.figure_dir = Path(figure_dir)
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot settings
        self.figsize_single = (12, 8)
        self.figsize_multi = (15, 10)
        self.dpi = 100
        
    def plot_from_analysis(self, analysis_results: Dict, results_df: pd.DataFrame = None):
        """
        Create plots based on analysis results.
        
        Args:
            analysis_results: Dictionary from ResultsAnalyzer
            results_df: Optional DataFrame with raw results
        """
        mode = analysis_results.get('mode', 'none')
        
        if mode == 'single':
            self._plot_single_parameter(analysis_results, results_df)
        elif mode == 'few':
            self._plot_few_parameters(analysis_results, results_df)
        elif mode == 'many':
            self._plot_many_parameters(analysis_results, results_df)
        else:
            print(f"No plots generated for mode: {mode}")
    
    def _plot_single_parameter(self, analysis_results: Dict, results_df: pd.DataFrame):
        """Create comprehensive plots for single parameter analysis."""
        param = analysis_results.get('parameter')
        if not param:
            return
        
        # Prepare data
        if results_df is not None:
            df = results_df.dropna(subset=[param, analysis_results.get('target_metric', 'trap_count')])
            x = df[param].values
            y = df[analysis_results.get('target_metric', 'trap_count')].values
            
            # Sort by parameter value
            sort_idx = np.argsort(x)
            x = x[sort_idx]
            y = y[sort_idx]
        else:
            print("No data available for plotting")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Main response curve
        ax1 = plt.subplot(2, 3, 1)
        self._plot_response_curve(ax1, x, y, param, analysis_results)
        
        # 2. Response curve with confidence intervals
        ax2 = plt.subplot(2, 3, 2)
        self._plot_response_with_confidence(ax2, x, y, param)
        
        # 3. Rate of change (derivative)
        ax3 = plt.subplot(2, 3, 3)
        self._plot_derivative(ax3, x, y, param)
        
        # 4. Distribution of outcomes
        ax4 = plt.subplot(2, 3, 4)
        self._plot_distribution(ax4, y, analysis_results.get('target_metric', 'trap_count'))
        
        # 5. Scatter with regression
        ax5 = plt.subplot(2, 3, 5)
        self._plot_scatter_with_fit(ax5, x, y, param, analysis_results)
        
        # 6. Summary statistics text
        ax6 = plt.subplot(2, 3, 6)
        self._plot_summary_text(ax6, analysis_results)
        
        plt.suptitle(f'Single Parameter Analysis: {param}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.figure_dir / f"single_param_{param}_{timestamp}.png"
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Single parameter plots saved to {filename}")
    
    def _plot_few_parameters(self, analysis_results: Dict, results_df: pd.DataFrame):
        """Create plots for few parameters analysis."""
        params = analysis_results.get('parameters', [])
        target_metric = analysis_results.get('target_metric', 'trap_count')
        
        if not params or results_df is None:
            return
        
        # Create figure
        n_params = len(params)
        fig = plt.figure(figsize=(16, 6 * ((n_params + 1) // 2)))
        
        # Plot response curves for each parameter
        for i, param in enumerate(params):
            ax = plt.subplot((n_params + 1) // 2, 2, i + 1)
            
            df_param = results_df.dropna(subset=[param, target_metric])
            if len(df_param) > 0:
                x = df_param[param].values
                y = df_param[target_metric].values
                
                # Sort by parameter value
                sort_idx = np.argsort(x)
                x = x[sort_idx]
                y = y[sort_idx]
                
                # Plot with smooth curve
                ax.scatter(x, y, alpha=0.6, s=50, label='Data points')
                
                if len(x) > 3:
                    # Fit smooth curve
                    x_smooth = np.linspace(x.min(), x.max(), 100)
                    f = interp1d(x, y, kind='cubic', fill_value='extrapolate')
                    y_smooth = f(x_smooth)
                    ax.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Fitted curve')
                
                ax.set_xlabel(param, fontsize=12)
                ax.set_ylabel(target_metric, fontsize=12)
                ax.set_title(f'{param} Response Curve', fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        plt.suptitle(f'Parameter Response Curves ({n_params} parameters)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.figure_dir / f"few_params_response_{timestamp}.png"
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        # Create interaction heatmap if available
        if 'pairwise_interactions' in analysis_results and analysis_results['pairwise_interactions']:
            self._plot_interaction_heatmap(analysis_results['pairwise_interactions'], params)
        
        # Create parameter ranking plot
        if 'parameter_ranking' in analysis_results:
            self._plot_parameter_ranking(analysis_results, 'few')
        
        print(f"Few parameters plots saved to {self.figure_dir}")
    
    def _plot_many_parameters(self, analysis_results: Dict, results_df: pd.DataFrame):
        """Create plots for many parameters analysis."""
        params = analysis_results.get('parameters', [])
        
        if not params:
            return
        
        # 1. Tornado plot for parameter importance
        if 'parameter_importance' in analysis_results:
            self._plot_tornado(analysis_results['parameter_importance'])
        
        # 2. Correlation heatmap
        if 'correlation_matrix' in analysis_results:
            self._plot_correlation_heatmap(analysis_results['correlation_matrix'])
        
        # 3. Parameter ranking bar chart
        if 'parameter_ranking' in analysis_results:
            self._plot_parameter_ranking(analysis_results, 'many')
        
        # 4. Box plots for each parameter
        if results_df is not None and len(params) <= 12:
            self._plot_parameter_boxplots(params, results_df, analysis_results.get('target_metric', 'trap_count'))
        
        print(f"Many parameters plots saved to {self.figure_dir}")
    
    def _plot_response_curve(self, ax, x, y, param, analysis_results):
        """Plot main response curve with optimal point."""
        ax.scatter(x, y, alpha=0.6, s=50, label='Simulations')
        
        # Add smooth curve if enough points
        if len(x) > 3:
            x_smooth = np.linspace(x.min(), x.max(), 200)
            
            # Use fitted polynomial if available
            if 'curve_fitting' in analysis_results:
                best_fit = analysis_results['curve_fitting'].get('best_fit', 'linear')
                coeffs = analysis_results['curve_fitting'].get(best_fit, {}).get('coefficients', [])
                if coeffs:
                    y_smooth = np.polyval(coeffs, x_smooth)
                    ax.plot(x_smooth, y_smooth, 'r-', linewidth=2, 
                           label=f'{best_fit.capitalize()} fit (R²={analysis_results["curve_fitting"][best_fit]["r2"]:.3f})')
        
        # Mark optimal point
        if 'optimal' in analysis_results:
            opt_x = analysis_results['optimal']['parameter_value']
            opt_y = analysis_results['optimal']['metric_value']
            ax.plot(opt_x, opt_y, 'g*', markersize=15, label=f'Optimal ({opt_x:.2f}, {opt_y:.0f})')
        
        ax.set_xlabel(param, fontsize=12)
        ax.set_ylabel(analysis_results.get('target_metric', 'trap_count'), fontsize=12)
        ax.set_title('Response Curve', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_response_with_confidence(self, ax, x, y, param):
        """Plot response curve with confidence intervals."""
        ax.scatter(x, y, alpha=0.4, s=30)
        
        # Calculate rolling mean and std
        window = max(3, len(x) // 5)
        if len(x) > window:
            df_temp = pd.DataFrame({'x': x, 'y': y}).sort_values('x')
            rolling_mean = df_temp['y'].rolling(window=window, center=True).mean()
            rolling_std = df_temp['y'].rolling(window=window, center=True).std()
            
            # Plot mean and confidence interval
            ax.plot(df_temp['x'], rolling_mean, 'b-', linewidth=2, label='Moving average')
            ax.fill_between(df_temp['x'], 
                           rolling_mean - 1.96 * rolling_std,
                           rolling_mean + 1.96 * rolling_std,
                           alpha=0.3, label='95% CI')
        
        ax.set_xlabel(param, fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title('Response with Confidence Intervals', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_derivative(self, ax, x, y, param):
        """Plot rate of change (derivative)."""
        if len(x) < 3:
            ax.text(0.5, 0.5, 'Insufficient data for derivative', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Calculate derivative
        dy_dx = np.gradient(y, x)
        
        # Plot
        ax.plot(x, dy_dx, 'o-', linewidth=2, markersize=6)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Mark maximum change point
        max_idx = np.argmax(np.abs(dy_dx))
        ax.plot(x[max_idx], dy_dx[max_idx], 'r*', markersize=12, 
               label=f'Max change at {x[max_idx]:.2f}')
        
        ax.set_xlabel(param, fontsize=12)
        ax.set_ylabel('Rate of Change', fontsize=12)
        ax.set_title('Derivative (Rate of Change)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_distribution(self, ax, y, metric_name):
        """Plot distribution of metric values."""
        ax.hist(y, bins=min(20, len(y)//2), edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(y), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(y):.1f}')
        ax.axvline(np.median(y), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(y):.1f}')
        
        ax.set_xlabel(metric_name, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Outcomes', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_scatter_with_fit(self, ax, x, y, param, analysis_results):
        """Plot scatter with regression line."""
        ax.scatter(x, y, alpha=0.6, s=50)
        
        # Add regression line
        if 'correlation' in analysis_results:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), 'r-', linewidth=2,
                   label=f"r = {analysis_results['correlation']['pearson_r']:.3f}\n"
                         f"p = {analysis_results['correlation']['p_value']:.3e}")
        
        ax.set_xlabel(param, fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title('Scatter with Linear Fit', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_summary_text(self, ax, analysis_results):
        """Plot summary statistics as text."""
        ax.axis('off')
        
        summary_text = "SUMMARY STATISTICS\n" + "="*30 + "\n\n"
        
        if 'statistics' in analysis_results:
            stats = analysis_results['statistics']
            summary_text += f"Parameter Range: [{stats['min_value']:.2f}, {stats['max_value']:.2f}]\n"
            summary_text += f"Metric Range: [{stats['min_metric']:.0f}, {stats['max_metric']:.0f}]\n"
            summary_text += f"Mean Metric: {stats['mean_metric']:.1f}\n"
            summary_text += f"Std Metric: {stats['std_metric']:.1f}\n"
            summary_text += f"Range Effect: {stats['range_effect']:.0f}\n\n"
        
        if 'optimal' in analysis_results:
            opt = analysis_results['optimal']
            summary_text += f"OPTIMAL POINT\n"
            summary_text += f"Parameter: {opt['parameter_value']:.2f}\n"
            summary_text += f"Metric: {opt['metric_value']:.0f}\n\n"
        
        if 'sensitivity' in analysis_results:
            sens = analysis_results['sensitivity']
            summary_text += f"SENSITIVITY\n"
            summary_text += f"Absolute: {sens['absolute']:.1f}\n"
            if sens.get('relative_to_baseline'):
                summary_text += f"Relative: {sens['relative_to_baseline']*100:.1f}%\n\n"
        
        if 'monotonicity' in analysis_results:
            mono = analysis_results['monotonicity']
            summary_text += f"MONOTONICITY\n"
            summary_text += f"Spearman r: {mono['spearman_r']:.3f}\n"
            summary_text += f"Is Monotonic: {mono['is_monotonic']}\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    def _plot_tornado(self, parameter_importance: Dict):
        """Create tornado plot for parameter importance."""
        # Prepare data
        params = []
        values = []
        for param, info in parameter_importance.items():
            params.append(param)
            values.append(abs(info.get('abs_correlation', 0)))
        
        # Sort by importance
        sorted_idx = np.argsort(values)
        params = [params[i] for i in sorted_idx]
        values = [values[i] for i in sorted_idx]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(params) * 0.4)))
        
        colors = ['red' if v > 0.5 else 'orange' if v > 0.3 else 'yellow' for v in values]
        bars = ax.barh(params, values, color=colors, edgecolor='black')
        
        ax.set_xlabel('Absolute Correlation with Target Metric', fontsize=12)
        ax.set_title('Parameter Importance (Tornado Plot)', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', va='center')
        
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.figure_dir / f"tornado_plot_{timestamp}.png"
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_heatmap(self, correlation_matrix: Dict):
        """Plot correlation heatmap."""
        # Convert dict to DataFrame
        df_corr = pd.DataFrame(correlation_matrix)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(df_corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, vmin=-1, vmax=1, square=True,
                   cbar_kws={"shrink": 0.8})
        
        ax.set_title('Parameter Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.figure_dir / f"correlation_heatmap_{timestamp}.png"
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_interaction_heatmap(self, interactions: Dict, params: List[str]):
        """Plot interaction effects heatmap."""
        # Create matrix
        n_params = len(params)
        interaction_matrix = np.zeros((n_params, n_params))
        
        for pair_key, interaction_info in interactions.items():
            param1, param2 = pair_key.split('-')
            if param1 in params and param2 in params:
                i = params.index(param1)
                j = params.index(param2)
                value = interaction_info.get('interaction_strength', 0)
                interaction_matrix[i, j] = value
                interaction_matrix[j, i] = value
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(interaction_matrix, annot=True, fmt='.3f', 
                   xticklabels=params, yticklabels=params,
                   cmap='YlOrRd', vmin=0, square=True,
                   cbar_kws={"label": "Interaction Strength"})
        
        ax.set_title('Parameter Interaction Effects', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.figure_dir / f"interaction_heatmap_{timestamp}.png"
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_ranking(self, analysis_results: Dict, mode: str):
        """Plot parameter ranking bar chart."""
        ranking = analysis_results.get('parameter_ranking', [])
        
        if not ranking:
            return
        
        # Get values for each parameter
        values = []
        if mode == 'few' and 'individual_effects' in analysis_results:
            effects = analysis_results['individual_effects']
            for param in ranking:
                values.append(effects.get(param, {}).get('range_effect', 0))
        elif mode == 'many' and 'parameter_importance' in analysis_results:
            importance = analysis_results['parameter_importance']
            for param in ranking:
                values.append(importance.get(param, {}).get('abs_correlation', 0))
        else:
            return
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(range(len(ranking)), values, color='steelblue', edgecolor='black')
        ax.set_xticks(range(len(ranking)))
        ax.set_xticklabels(ranking, rotation=45, ha='right')
        
        if mode == 'few':
            ax.set_ylabel('Range Effect on Target Metric', fontsize=12)
        else:
            ax.set_ylabel('Absolute Correlation', fontsize=12)
        
        ax.set_title('Parameter Ranking by Impact', fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value + max(values)*0.01, 
                   f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.figure_dir / f"parameter_ranking_{mode}_{timestamp}.png"
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_boxplots(self, params: List[str], results_df: pd.DataFrame, target_metric: str):
        """Create box plots for multiple parameters."""
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, param in enumerate(params[:12]):  # Limit to 12 parameters
            ax = axes[i]
            
            df_param = results_df.dropna(subset=[param, target_metric])
            if len(df_param) > 0:
                # Create bins for parameter values
                param_values = df_param[param].values
                n_bins = min(5, len(np.unique(param_values)))
                
                if n_bins > 1:
                    bins = pd.qcut(param_values, n_bins, duplicates='drop')
                    df_param['bin'] = bins
                    
                    # Create box plot
                    df_param.boxplot(column=target_metric, by='bin', ax=ax)
                    ax.set_xlabel(param, fontsize=10)
                    ax.set_ylabel(target_metric, fontsize=10)
                    ax.set_title('')
                    ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(params), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Parameter Effects on Target Metric (Box Plots)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.figure_dir / f"parameter_boxplots_{timestamp}.png"
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def create_summary_plot(self, results_df: pd.DataFrame, target_metric: str = "trap_count"):
        """Create a summary plot showing all simulation results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Distribution of target metric
        ax = axes[0, 0]
        ax.hist(results_df[target_metric], bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(results_df[target_metric].mean(), color='r', linestyle='--', 
                  label=f'Mean: {results_df[target_metric].mean():.1f}')
        ax.set_xlabel(target_metric, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Distribution of {target_metric}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Simulation index vs metric
        ax = axes[0, 1]
        ax.plot(results_df.index, results_df[target_metric], 'o-', alpha=0.7)
        ax.set_xlabel('Simulation Index', fontsize=12)
        ax.set_ylabel(target_metric, fontsize=12)
        ax.set_title('Simulation Results Over Time', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 3. Execution time distribution
        if 'execution_time' in results_df.columns:
            ax = axes[1, 0]
            ax.hist(results_df['execution_time'], bins=20, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Execution Time (s)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Execution Time Distribution', fontsize=14)
            ax.grid(True, alpha=0.3)
        else:
            axes[1, 0].axis('off')
        
        # 4. Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = f"SUMMARY STATISTICS\n{'='*30}\n\n"
        summary_text += f"Total Simulations: {len(results_df)}\n"
        summary_text += f"\n{target_metric}:\n"
        summary_text += f"  Mean: {results_df[target_metric].mean():.2f}\n"
        summary_text += f"  Std: {results_df[target_metric].std():.2f}\n"
        summary_text += f"  Min: {results_df[target_metric].min():.2f}\n"
        summary_text += f"  Max: {results_df[target_metric].max():.2f}\n"
        summary_text += f"  Range: {results_df[target_metric].max() - results_df[target_metric].min():.2f}\n"
        
        if 'execution_time' in results_df.columns:
            summary_text += f"\nExecution Time:\n"
            summary_text += f"  Mean: {results_df['execution_time'].mean():.2f}s\n"
            summary_text += f"  Total: {results_df['execution_time'].sum():.2f}s\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('Simulation Results Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.figure_dir / f"summary_plot_{timestamp}.png"
        plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Summary plot saved to {filename}")


def main():
    """Main function for testing the plotter."""
    from results_analyzer import ResultsAnalyzer
    
    # Load and analyze results
    analyzer = ResultsAnalyzer()
    df = analyzer.load_results()
    analysis_results = analyzer.analyze()
    
    # Create plots
    plotter = SensitivityPlotter()
    plotter.plot_from_analysis(analysis_results, df)
    plotter.create_summary_plot(df)
    
    print("Plotting complete!")


if __name__ == "__main__":
    main()