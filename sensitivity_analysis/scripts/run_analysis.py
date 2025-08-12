#!/usr/bin/env python3
"""
Master Analysis Orchestrator for SMOL Sensitivity Analysis

This script coordinates all analysis components and generates comprehensive
reports. It automatically detects the analysis mode and runs appropriate
analyses based on the available data.

Usage:
    python run_analysis.py [options]
    
Options:
    --mode auto|single|few|many  Analysis mode (default: auto)
    --metric METRIC              Target metric (default: trap_count)
    --parameters PARAM1 PARAM2   Specific parameters to analyze
    --output-dir DIR            Output directory (default: ../data/output)
    --generate-report           Generate HTML report
    --no-plots                  Skip plot generation
    
Author: SMOL Sensitivity Analysis Framework
Date: 2025
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from results_analyzer import ResultsAnalyzer
from sensitivity_plots import SensitivityPlotter
from correlation_analysis import CorrelationAnalyzer
from report_generator import ReportGenerator


class AnalysisOrchestrator:
    """
    Master orchestrator for sensitivity analysis.
    
    Coordinates all analysis components and generates comprehensive reports.
    """
    
    def __init__(self, output_dir: str = "../data/output", 
                 figure_dir: str = "../figures",
                 metric: str = "trap_count"):
        """
        Initialize the orchestrator.
        
        Args:
            output_dir: Directory containing simulation outputs
            figure_dir: Directory to save figures
            metric: Target metric to analyze
        """
        self.output_dir = Path(output_dir)
        self.figure_dir = Path(figure_dir)
        self.metric = metric
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize components
        self.analyzer = ResultsAnalyzer(str(self.output_dir))
        self.plotter = SensitivityPlotter(str(self.output_dir), str(self.figure_dir))
        self.corr_analyzer = CorrelationAnalyzer(str(self.figure_dir))
        self.report_generator = ReportGenerator(str(self.output_dir), str(self.figure_dir))
        
        # Results storage
        self.results = {
            'timestamp': self.timestamp,
            'metric': metric,
            'components': {}
        }
    
    def run(self, mode: str = "auto", parameters: Optional[List[str]] = None,
            generate_plots: bool = True, generate_report: bool = False) -> Dict:
        """
        Run complete analysis pipeline.
        
        Args:
            mode: Analysis mode (auto, single, few, many)
            parameters: Specific parameters to analyze (None = all)
            generate_plots: Whether to generate plots
            generate_report: Whether to generate HTML report
            
        Returns:
            Dictionary containing all analysis results
        """
        print(f"\n{'='*60}")
        print("SMOL SENSITIVITY ANALYSIS ORCHESTRATOR")
        print(f"{'='*60}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Target metric: {self.metric}")
        print(f"Mode: {mode}")
        print()
        
        # Step 1: Load data
        print("Step 1: Loading simulation results...")
        df = self.analyzer.load_results()
        
        if df.empty:
            print("Error: No simulation results found!")
            return self.results
        
        print(f"  Loaded {len(df)} simulations")
        self.results['n_simulations'] = len(df)
        
        # Filter to specific parameters if requested
        if parameters:
            # Verify parameters exist
            valid_params = [p for p in parameters if p in df.columns]
            if not valid_params:
                print(f"Error: None of the specified parameters found: {parameters}")
                return self.results
            
            if len(valid_params) < len(parameters):
                print(f"Warning: Some parameters not found. Using: {valid_params}")
            
            # Override detected parameters
            self.analyzer.varied_parameters = valid_params
            self.analyzer._determine_analysis_mode()
        
        # Override mode if specified
        if mode != "auto":
            if mode in ["single", "few", "many"]:
                self.analyzer.analysis_mode = mode
                print(f"  Using specified mode: {mode}")
            else:
                print(f"Warning: Invalid mode '{mode}'. Using auto-detection.")
        
        self.results['mode'] = self.analyzer.analysis_mode
        self.results['parameters'] = self.analyzer.varied_parameters
        
        print(f"  Analysis mode: {self.analyzer.analysis_mode}")
        print(f"  Varied parameters: {len(self.analyzer.varied_parameters)}")
        
        if self.analyzer.varied_parameters:
            print(f"    {', '.join(self.analyzer.varied_parameters)}")
        
        # Step 2: Core analysis
        print("\nStep 2: Performing core analysis...")
        analysis_results = self.analyzer.analyze(self.metric)
        self.results['components']['core_analysis'] = analysis_results
        
        # Print key findings
        self._print_key_findings(analysis_results)
        
        # Step 3: Correlation analysis (if multiple parameters)
        if len(self.analyzer.varied_parameters) > 1:
            print("\nStep 3: Performing correlation analysis...")
            correlation_results = self.corr_analyzer.analyze_correlations(
                df, self.analyzer.varied_parameters, self.metric
            )
            self.results['components']['correlation_analysis'] = correlation_results
            
            # Print correlation summary
            self._print_correlation_summary(correlation_results)
        else:
            print("\nStep 3: Skipping correlation analysis (single parameter)")
        
        # Step 4: Generate visualizations
        if generate_plots:
            print("\nStep 4: Generating visualizations...")
            self.plotter.plot_from_analysis(analysis_results, df)
            self.plotter.create_summary_plot(df, self.metric)
            print(f"  Plots saved to {self.figure_dir}")
        else:
            print("\nStep 4: Skipping visualization (--no-plots)")
        
        # Step 5: Generate summary statistics
        print("\nStep 5: Computing summary statistics...")
        summary_stats = self._compute_summary_statistics(df)
        self.results['components']['summary_statistics'] = summary_stats
        self._print_summary_statistics(summary_stats)
        
        # Step 6: Generate recommendations
        print("\nStep 6: Generating recommendations...")
        recommendations = self._generate_recommendations(analysis_results)
        self.results['recommendations'] = recommendations
        self._print_recommendations(recommendations)
        
        # Step 7: Save results
        print("\nStep 7: Saving analysis results...")
        self._save_results()
        
        # Step 8: Generate HTML report
        if generate_report:
            print("\nStep 8: Generating HTML report...")
            report_path = self.report_generator.generate_report(
                self.results, df, self.timestamp
            )
            print(f"  Report saved to {report_path}")
        else:
            print("\nStep 8: Skipping HTML report generation")
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        
        return self.results
    
    def _print_key_findings(self, analysis_results: Dict):
        """Print key findings from core analysis."""
        mode = analysis_results.get('mode', 'unknown')
        
        if mode == 'single':
            param = analysis_results.get('parameter')
            if 'optimal' in analysis_results:
                opt = analysis_results['optimal']
                print(f"  Optimal {param}: {opt['parameter_value']:.2f} → {self.metric}: {opt['metric_value']:.0f}")
            
            if 'correlation' in analysis_results:
                corr = analysis_results['correlation']
                print(f"  Correlation: r={corr['pearson_r']:.3f} (p={corr['p_value']:.3e})")
            
            if 'sensitivity' in analysis_results:
                sens = analysis_results['sensitivity']
                print(f"  Sensitivity range: {sens['absolute']:.0f}")
        
        elif mode == 'few':
            if 'parameter_ranking' in analysis_results:
                ranking = analysis_results['parameter_ranking']
                print(f"  Top 3 parameters: {', '.join(ranking[:3])}")
            
            if 'pairwise_interactions' in analysis_results:
                interactions = analysis_results['pairwise_interactions']
                significant = [k for k, v in interactions.items() 
                             if v.get('has_significant_interaction', False)]
                if significant:
                    print(f"  Significant interactions: {len(significant)}")
        
        elif mode == 'many':
            if 'parameter_ranking' in analysis_results:
                ranking = analysis_results['parameter_ranking']
                print(f"  Top 5 parameters: {', '.join(ranking[:5])}")
            
            if 'summary_statistics' in analysis_results:
                stats = analysis_results['summary_statistics']
                print(f"  {self.metric} range: [{stats['metric_min']:.0f}, {stats['metric_max']:.0f}]")
    
    def _print_correlation_summary(self, correlation_results: Dict):
        """Print correlation analysis summary."""
        if 'pearson' in correlation_results:
            significant = correlation_results['pearson'].get('significant', [])
            if significant:
                print(f"  Significant correlations (p<0.05): {', '.join(significant)}")
        
        if 'pca' in correlation_results:
            pca = correlation_results['pca']
            print(f"  PCA: {pca['n_components_95_variance']} components for 95% variance")
        
        if 'clustering' in correlation_results:
            clustering = correlation_results['clustering']
            print(f"  Parameter clusters: {clustering['optimal_clusters']}")
    
    def _compute_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """Compute comprehensive summary statistics."""
        stats = {
            'total_simulations': len(df),
            'successful_simulations': len(df[df.get('status', 'completed') == 'completed'])
                if 'status' in df.columns else len(df),
            'metric_statistics': {
                'mean': float(df[self.metric].mean()),
                'std': float(df[self.metric].std()),
                'min': float(df[self.metric].min()),
                'max': float(df[self.metric].max()),
                'median': float(df[self.metric].median()),
                'q25': float(df[self.metric].quantile(0.25)),
                'q75': float(df[self.metric].quantile(0.75))
            }
        }
        
        # Parameter statistics
        if self.analyzer.varied_parameters:
            param_stats = {}
            for param in self.analyzer.varied_parameters:
                if param in df.columns:
                    param_stats[param] = {
                        'min': float(df[param].min()),
                        'max': float(df[param].max()),
                        'unique_values': int(df[param].nunique())
                    }
            stats['parameter_statistics'] = param_stats
        
        # Execution time statistics
        if 'execution_time' in df.columns:
            stats['execution_statistics'] = {
                'total_time': float(df['execution_time'].sum()),
                'mean_time': float(df['execution_time'].mean()),
                'max_time': float(df['execution_time'].max())
            }
        
        return stats
    
    def _print_summary_statistics(self, stats: Dict):
        """Print summary statistics."""
        metric_stats = stats['metric_statistics']
        print(f"  {self.metric}: mean={metric_stats['mean']:.1f}, "
              f"std={metric_stats['std']:.1f}, "
              f"range=[{metric_stats['min']:.0f}, {metric_stats['max']:.0f}]")
        
        if 'execution_statistics' in stats:
            exec_stats = stats['execution_statistics']
            print(f"  Execution: total={exec_stats['total_time']:.1f}s, "
                  f"mean={exec_stats['mean_time']:.1f}s")
    
    def _generate_recommendations(self, analysis_results: Dict) -> List[Dict]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        mode = analysis_results.get('mode', 'unknown')
        
        if mode == 'single':
            # Single parameter recommendations
            if 'optimal' in analysis_results:
                opt = analysis_results['optimal']
                param = analysis_results.get('parameter')
                recommendations.append({
                    'type': 'optimal_value',
                    'priority': 'high',
                    'message': f"Set {param} to {opt['parameter_value']:.2f} for optimal {self.metric} ({opt['metric_value']:.0f})"
                })
            
            if 'sensitivity' in analysis_results:
                sens = analysis_results['sensitivity']
                if sens.get('relative_to_baseline') and sens['relative_to_baseline'] > 0.2:
                    recommendations.append({
                        'type': 'high_sensitivity',
                        'priority': 'high',
                        'message': f"Parameter shows high sensitivity ({sens['relative_to_baseline']*100:.1f}% variation). Careful tuning recommended."
                    })
        
        elif mode in ['few', 'many']:
            # Multiple parameter recommendations
            if 'parameter_ranking' in analysis_results:
                ranking = analysis_results['parameter_ranking']
                if ranking:
                    top_params = ranking[:3]
                    recommendations.append({
                        'type': 'focus_parameters',
                        'priority': 'high',
                        'message': f"Focus optimization on top parameters: {', '.join(top_params)}"
                    })
            
            # Check for interactions
            if 'pairwise_interactions' in analysis_results:
                interactions = analysis_results['pairwise_interactions']
                significant = [k for k, v in interactions.items() 
                             if v.get('has_significant_interaction', False)]
                if significant:
                    recommendations.append({
                        'type': 'parameter_interactions',
                        'priority': 'medium',
                        'message': f"Consider joint optimization for interacting parameters: {', '.join(significant)}"
                    })
        
        # General recommendations
        if 'correlation' in analysis_results:
            corr = analysis_results.get('correlation', {})
            if corr.get('is_monotonic'):
                direction = "increasing" if corr.get('spearman_r', 0) > 0 else "decreasing"
                recommendations.append({
                    'type': 'monotonic_relationship',
                    'priority': 'medium',
                    'message': f"Parameter shows monotonic {direction} relationship. Consider boundary values."
                })
        
        # Add recommendation for further analysis if needed
        if len(self.analyzer.varied_parameters) == 1:
            recommendations.append({
                'type': 'further_analysis',
                'priority': 'low',
                'message': "Consider multi-parameter analysis to identify interactions and combined effects"
            })
        
        return recommendations
    
    def _print_recommendations(self, recommendations: List[Dict]):
        """Print recommendations."""
        if not recommendations:
            print("  No specific recommendations generated")
            return
        
        # Group by priority
        high = [r for r in recommendations if r['priority'] == 'high']
        medium = [r for r in recommendations if r['priority'] == 'medium']
        low = [r for r in recommendations if r['priority'] == 'low']
        
        if high:
            print("  High Priority:")
            for r in high:
                print(f"    • {r['message']}")
        
        if medium:
            print("  Medium Priority:")
            for r in medium:
                print(f"    • {r['message']}")
        
        if low:
            print("  Low Priority:")
            for r in low:
                print(f"    • {r['message']}")
    
    def _save_results(self):
        """Save all analysis results to JSON."""
        output_file = self.output_dir / f"complete_analysis_{self.timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"  Results saved to {output_file}")
        
        # Also save summary CSV
        if 'components' in self.results and 'summary_statistics' in self.results['components']:
            summary_file = self.output_dir / f"analysis_summary_{self.timestamp}.csv"
            
            # Create summary DataFrame
            summary_data = []
            
            # Add parameter statistics
            if 'parameter_statistics' in self.results['components']['summary_statistics']:
                for param, stats in self.results['components']['summary_statistics']['parameter_statistics'].items():
                    summary_data.append({
                        'type': 'parameter',
                        'name': param,
                        'min': stats['min'],
                        'max': stats['max'],
                        'unique_values': stats['unique_values']
                    })
            
            # Add metric statistics
            if 'metric_statistics' in self.results['components']['summary_statistics']:
                m_stats = self.results['components']['summary_statistics']['metric_statistics']
                summary_data.append({
                    'type': 'metric',
                    'name': self.metric,
                    'min': m_stats['min'],
                    'max': m_stats['max'],
                    'mean': m_stats['mean'],
                    'std': m_stats['std']
                })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_csv(summary_file, index=False)
                print(f"  Summary saved to {summary_file}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Master orchestrator for SMOL sensitivity analysis'
    )
    
    parser.add_argument('--mode', choices=['auto', 'single', 'few', 'many'],
                       default='auto', help='Analysis mode (default: auto)')
    parser.add_argument('--metric', default='trap_count',
                       help='Target metric to analyze (default: trap_count)')
    parser.add_argument('--parameters', nargs='+',
                       help='Specific parameters to analyze')
    parser.add_argument('--output-dir', default='../data/output',
                       help='Output directory (default: ../data/output)')
    parser.add_argument('--figure-dir', default='../figures',
                       help='Figure directory (default: ../figures)')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate HTML report')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = AnalysisOrchestrator(
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        metric=args.metric
    )
    
    # Run analysis
    results = orchestrator.run(
        mode=args.mode,
        parameters=args.parameters,
        generate_plots=not args.no_plots,
        generate_report=args.generate_report
    )
    
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())