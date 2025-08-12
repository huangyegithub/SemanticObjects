#!/usr/bin/env python3
"""
HTML Report Generator for SMOL Sensitivity Analysis

This module generates comprehensive HTML reports from sensitivity analysis results.
Reports are adaptive based on the analysis mode (single, few, many parameters).

Author: SMOL Sensitivity Analysis Framework
Date: 2025
"""

import json
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


class ReportGenerator:
    """
    Generates adaptive HTML reports for sensitivity analysis results.
    """
    
    def __init__(self, output_dir: str = "../data/output", 
                 figure_dir: str = "../figures"):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory for output files
            figure_dir: Directory containing figures
        """
        self.output_dir = Path(output_dir)
        self.figure_dir = Path(figure_dir)
        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(self, analysis_results: Dict, 
                       results_df: pd.DataFrame,
                       timestamp: str = None) -> Path:
        """
        Generate comprehensive HTML report.
        
        Args:
            analysis_results: Complete analysis results dictionary
            results_df: DataFrame with simulation results
            timestamp: Timestamp for the report
            
        Returns:
            Path to the generated HTML report
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        mode = analysis_results.get('mode', 'unknown')
        
        # Generate appropriate report based on mode
        if mode == 'single':
            html_content = self._generate_single_parameter_report(analysis_results, results_df)
        elif mode == 'few':
            html_content = self._generate_few_parameters_report(analysis_results, results_df)
        elif mode == 'many':
            html_content = self._generate_many_parameters_report(analysis_results, results_df)
        else:
            html_content = self._generate_basic_report(analysis_results, results_df)
        
        # Save report
        report_path = self.reports_dir / f"sensitivity_report_{mode}_{timestamp}.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return report_path
    
    def _generate_single_parameter_report(self, analysis_results: Dict, 
                                         results_df: pd.DataFrame) -> str:
        """Generate report for single parameter analysis."""
        core_analysis = analysis_results.get('components', {}).get('core_analysis', {})
        parameter = core_analysis.get('parameter', 'Unknown')
        metric = analysis_results.get('metric', 'trap_count')
        
        html = self._get_html_header("Single Parameter Sensitivity Analysis")
        
        # Executive Summary
        html += """
        <section class="summary">
            <h2>Executive Summary</h2>
            <div class="summary-content">
        """
        
        if 'optimal' in core_analysis:
            opt = core_analysis['optimal']
            html += f"""
                <div class="key-finding">
                    <h3>Optimal Configuration</h3>
                    <p><strong>{parameter}</strong> = {opt['parameter_value']:.2f}</p>
                    <p><strong>{metric}</strong> = {opt['metric_value']:.0f}</p>
                </div>
            """
        
        if 'sensitivity' in core_analysis:
            sens = core_analysis['sensitivity']
            html += f"""
                <div class="key-finding">
                    <h3>Sensitivity</h3>
                    <p>Absolute Range: {sens['absolute']:.0f}</p>
                    {'<p>Relative: ' + f"{sens['relative_to_baseline']*100:.1f}%" + '</p>' 
                     if sens.get('relative_to_baseline') else ''}
                </div>
            """
        
        if 'correlation' in core_analysis:
            corr = core_analysis['correlation']
            html += f"""
                <div class="key-finding">
                    <h3>Correlation</h3>
                    <p>Pearson r: {corr['pearson_r']:.3f}</p>
                    <p>Relationship: {corr.get('relationship', 'Unknown')}</p>
                </div>
            """
        
        html += """
            </div>
        </section>
        """
        
        # Detailed Analysis
        html += """
        <section class="analysis">
            <h2>Detailed Analysis</h2>
        """
        
        # Statistics table
        if 'statistics' in core_analysis:
            stats = core_analysis['statistics']
            html += self._create_statistics_table(stats, parameter, metric)
        
        # Curve fitting results
        if 'curve_fitting' in core_analysis and 'best_fit' in core_analysis['curve_fitting']:
            best_fit = core_analysis['curve_fitting']['best_fit']
            html += f"""
            <div class="subsection">
                <h3>Best Fit Model</h3>
                <p><strong>Model:</strong> {best_fit['model']}</p>
                <p><strong>R² Score:</strong> {best_fit['r2']:.4f}</p>
                <p><strong>Equation:</strong> <code>{best_fit['equation']}</code></p>
            </div>
            """
        
        # Critical points
        if 'critical_points' in core_analysis:
            html += self._create_critical_points_section(core_analysis['critical_points'])
        
        html += "</section>"
        
        # Visualizations
        html += self._add_visualizations_section(parameter)
        
        # Recommendations
        if 'recommendations' in analysis_results:
            html += self._create_recommendations_section(analysis_results['recommendations'])
        
        # Data table
        html += self._create_data_table(results_df, [parameter], metric)
        
        html += self._get_html_footer()
        
        return html
    
    def _generate_few_parameters_report(self, analysis_results: Dict, 
                                       results_df: pd.DataFrame) -> str:
        """Generate report for few parameters analysis."""
        core_analysis = analysis_results.get('components', {}).get('core_analysis', {})
        parameters = core_analysis.get('parameters', [])
        metric = analysis_results.get('metric', 'trap_count')
        
        html = self._get_html_header("Multi-Parameter Sensitivity Analysis")
        
        # Executive Summary
        html += """
        <section class="summary">
            <h2>Executive Summary</h2>
            <div class="summary-content">
        """
        
        # Parameter ranking
        if 'parameter_ranking' in core_analysis:
            ranking = core_analysis['parameter_ranking']
            html += """
                <div class="key-finding">
                    <h3>Parameter Importance Ranking</h3>
                    <ol>
            """
            for i, param in enumerate(ranking[:5], 1):
                html += f"<li>{param}</li>"
            html += """
                    </ol>
                </div>
            """
        
        # Interactions
        if 'pairwise_interactions' in core_analysis:
            interactions = core_analysis['pairwise_interactions']
            significant = [k for k, v in interactions.items() 
                         if v.get('has_significant_interaction', False)]
            if significant:
                html += f"""
                <div class="key-finding">
                    <h3>Parameter Interactions</h3>
                    <p>{len(significant)} significant interaction(s) detected</p>
                    <p>Pairs: {', '.join(significant)}</p>
                </div>
                """
        
        html += """
            </div>
        </section>
        """
        
        # Individual Effects
        html += """
        <section class="analysis">
            <h2>Individual Parameter Effects</h2>
        """
        
        if 'individual_effects' in core_analysis:
            html += self._create_individual_effects_table(core_analysis['individual_effects'])
        
        html += "</section>"
        
        # Correlation Analysis
        if 'correlation_analysis' in analysis_results.get('components', {}):
            html += self._add_correlation_section(
                analysis_results['components']['correlation_analysis']
            )
        
        # Visualizations
        html += self._add_visualizations_section("multi_parameter")
        
        # Recommendations
        if 'recommendations' in analysis_results:
            html += self._create_recommendations_section(analysis_results['recommendations'])
        
        # Data table
        html += self._create_data_table(results_df, parameters, metric)
        
        html += self._get_html_footer()
        
        return html
    
    def _generate_many_parameters_report(self, analysis_results: Dict, 
                                        results_df: pd.DataFrame) -> str:
        """Generate report for many parameters analysis."""
        core_analysis = analysis_results.get('components', {}).get('core_analysis', {})
        parameters = core_analysis.get('parameters', [])
        metric = analysis_results.get('metric', 'trap_count')
        
        html = self._get_html_header("Comprehensive Sensitivity Analysis")
        
        # Executive Summary
        html += """
        <section class="summary">
            <h2>Executive Summary</h2>
            <div class="summary-content">
        """
        
        # Summary statistics
        if 'summary_statistics' in core_analysis:
            stats = core_analysis['summary_statistics']
            html += f"""
                <div class="key-finding">
                    <h3>Analysis Overview</h3>
                    <p>Parameters analyzed: {len(parameters)}</p>
                    <p>Simulations: {stats['n_simulations']}</p>
                    <p>{metric} range: [{stats['metric_min']:.0f}, {stats['metric_max']:.0f}]</p>
                    <p>Mean ± Std: {stats['metric_mean']:.1f} ± {stats['metric_std']:.1f}</p>
                </div>
            """
        
        # Top parameters
        if 'parameter_ranking' in core_analysis:
            ranking = core_analysis['parameter_ranking']
            html += """
                <div class="key-finding">
                    <h3>Most Influential Parameters</h3>
                    <ol>
            """
            for param in ranking[:5]:
                if 'parameter_importance' in core_analysis:
                    corr = core_analysis['parameter_importance'].get(param, {}).get('abs_correlation', 0)
                    html += f"<li>{param} (|r| = {corr:.3f})</li>"
                else:
                    html += f"<li>{param}</li>"
            html += """
                    </ol>
                </div>
            """
        
        html += """
            </div>
        </section>
        """
        
        # Parameter Importance
        html += """
        <section class="analysis">
            <h2>Parameter Importance Analysis</h2>
        """
        
        if 'parameter_importance' in core_analysis:
            html += self._create_importance_table(core_analysis['parameter_importance'])
        
        # Sensitivity indices if available
        if 'sensitivity_indices' in core_analysis:
            html += self._create_sensitivity_indices_section(core_analysis['sensitivity_indices'])
        
        html += "</section>"
        
        # Correlation Analysis
        if 'correlation_analysis' in analysis_results.get('components', {}):
            html += self._add_correlation_section(
                analysis_results['components']['correlation_analysis']
            )
        
        # Visualizations
        html += self._add_visualizations_section("comprehensive")
        
        # Recommendations
        if 'recommendations' in analysis_results:
            html += self._create_recommendations_section(analysis_results['recommendations'])
        
        # Summary Statistics
        if 'summary_statistics' in analysis_results.get('components', {}):
            html += self._create_summary_statistics_section(
                analysis_results['components']['summary_statistics']
            )
        
        html += self._get_html_footer()
        
        return html
    
    def _generate_basic_report(self, analysis_results: Dict, 
                              results_df: pd.DataFrame) -> str:
        """Generate basic report when mode is unknown."""
        html = self._get_html_header("Sensitivity Analysis Report")
        
        html += """
        <section class="summary">
            <h2>Analysis Results</h2>
            <div class="summary-content">
                <p>Analysis mode could not be determined.</p>
                <p>Displaying available results below.</p>
            </div>
        </section>
        """
        
        # Display raw results as JSON
        html += """
        <section class="analysis">
            <h2>Raw Results</h2>
            <pre><code>
        """
        html += json.dumps(analysis_results, indent=2)
        html += """
            </code></pre>
        </section>
        """
        
        html += self._get_html_footer()
        
        return html
    
    def _get_html_header(self, title: str) -> str:
        """Get HTML header with styles."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - SMOL Analysis</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 0;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        section {{
            background: white;
            margin: 20px 0;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        h2 {{
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #f0f0f0;
        }}
        
        h3 {{
            color: #444;
            margin: 15px 0;
        }}
        
        .summary-content {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }}
        
        .key-finding {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        
        .key-finding h3 {{
            margin-top: 0;
            color: #667eea;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        th {{
            background-color: #667eea;
            color: white;
            font-weight: 600;
        }}
        
        tr:hover {{
            background-color: #f5f5f5;
        }}
        
        .visualization {{
            margin: 20px 0;
            text-align: center;
        }}
        
        .visualization img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .recommendations {{
            background: #e8f5e9;
            border-left: 4px solid #4caf50;
        }}
        
        .recommendations ul {{
            list-style-position: inside;
            margin-left: 20px;
        }}
        
        .recommendations li {{
            margin: 10px 0;
        }}
        
        .priority-high {{
            color: #d32f2f;
            font-weight: bold;
        }}
        
        .priority-medium {{
            color: #f57c00;
            font-weight: bold;
        }}
        
        .priority-low {{
            color: #388e3c;
            font-weight: bold;
        }}
        
        pre {{
            background: #f5f5f5;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
        }}
        
        code {{
            font-family: 'Courier New', Courier, monospace;
            background: #f5f5f5;
            padding: 2px 6px;
            border-radius: 3px;
        }}
        
        .subsection {{
            margin: 20px 0;
            padding: 15px;
            background: #fafafa;
            border-radius: 4px;
        }}
        
        footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
        
        .data-table {{
            max-height: 400px;
            overflow-y: auto;
        }}
        
        @media print {{
            header {{
                background: none;
                color: black;
            }}
            
            section {{
                box-shadow: none;
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>{title}</h1>
            <p>SMOL Geological Simulation - Sensitivity Analysis Framework</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </header>
    <div class="container">
"""
    
    def _get_html_footer(self) -> str:
        """Get HTML footer."""
        return """
    </div>
    <footer>
        <p>© 2025 SMOL Sensitivity Analysis Framework</p>
        <p>Report generated automatically by report_generator.py</p>
    </footer>
</body>
</html>
"""
    
    def _create_statistics_table(self, stats: Dict, parameter: str, metric: str) -> str:
        """Create statistics table HTML."""
        html = """
        <div class="subsection">
            <h3>Statistical Summary</h3>
            <table>
                <thead>
                    <tr>
                        <th>Statistic</th>
                        <th>Parameter Value</th>
                        <th>Metric Value</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        html += f"""
            <tr>
                <td>Minimum</td>
                <td>{stats['min_value']:.2f}</td>
                <td>{stats['min_metric']:.0f}</td>
            </tr>
            <tr>
                <td>Maximum</td>
                <td>{stats['max_value']:.2f}</td>
                <td>{stats['max_metric']:.0f}</td>
            </tr>
            <tr>
                <td>Mean</td>
                <td>-</td>
                <td>{stats['mean_metric']:.1f}</td>
            </tr>
            <tr>
                <td>Std Dev</td>
                <td>-</td>
                <td>{stats['std_metric']:.1f}</td>
            </tr>
            <tr>
                <td>Range</td>
                <td>{stats['max_value'] - stats['min_value']:.2f}</td>
                <td>{stats['range_effect']:.0f}</td>
            </tr>
        """
        
        html += """
                </tbody>
            </table>
        </div>
        """
        
        return html
    
    def _create_critical_points_section(self, critical_points: Dict) -> str:
        """Create critical points section HTML."""
        html = """
        <div class="subsection">
            <h3>Critical Points</h3>
        """
        
        if 'steepest_increase' in critical_points:
            point = critical_points['steepest_increase']
            html += f"""
            <p><strong>Steepest Increase:</strong> 
               Parameter = {point['parameter_value']:.2f}, 
               Gradient = {point['gradient']:.2f}</p>
            """
        
        if 'plateau_region' in critical_points:
            plateau = critical_points['plateau_region']
            html += f"""
            <p><strong>Plateau Region:</strong> 
               [{plateau['start']:.2f}, {plateau['end']:.2f}], 
               Average = {plateau['average_metric']:.1f}</p>
            """
        
        html += "</div>"
        
        return html
    
    def _create_individual_effects_table(self, individual_effects: Dict) -> str:
        """Create individual effects table HTML."""
        html = """
        <table>
            <thead>
                <tr>
                    <th>Parameter</th>
                    <th>Correlation</th>
                    <th>P-value</th>
                    <th>Range Effect</th>
                    <th>Mean Effect</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for param, effects in individual_effects.items():
            html += f"""
            <tr>
                <td>{param}</td>
                <td>{effects.get('correlation', 0):.3f}</td>
                <td>{effects.get('p_value', 1):.3e}</td>
                <td>{effects.get('range_effect', 0):.0f}</td>
                <td>{effects.get('mean_effect', 0):.1f}</td>
            </tr>
            """
        
        html += """
            </tbody>
        </table>
        """
        
        return html
    
    def _create_importance_table(self, parameter_importance: Dict) -> str:
        """Create parameter importance table HTML."""
        # Sort by absolute correlation
        sorted_params = sorted(parameter_importance.items(), 
                             key=lambda x: x[1].get('abs_correlation', 0), 
                             reverse=True)
        
        html = """
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Parameter</th>
                    <th>Correlation</th>
                    <th>Absolute Correlation</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for i, (param, importance) in enumerate(sorted_params, 1):
            html += f"""
            <tr>
                <td>{i}</td>
                <td>{param}</td>
                <td>{importance.get('correlation', 0):.3f}</td>
                <td>{importance.get('abs_correlation', 0):.3f}</td>
            </tr>
            """
        
        html += """
            </tbody>
        </table>
        """
        
        return html
    
    def _create_sensitivity_indices_section(self, sensitivity_indices: Dict) -> str:
        """Create sensitivity indices section HTML."""
        html = """
        <div class="subsection">
            <h3>Sensitivity Indices</h3>
        """
        
        if 'method' in sensitivity_indices:
            html += f"<p><strong>Method:</strong> {sensitivity_indices['method']}</p>"
        
        if 'indices' in sensitivity_indices:
            html += """
            <table>
                <thead>
                    <tr>
                        <th>Parameter</th>
                        <th>First Order Index</th>
                        <th>Total Order Index</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for param, indices in sensitivity_indices['indices'].items():
                first_order = indices.get('first_order', 'N/A')
                total_order = indices.get('total_order', 'N/A')
                
                html += f"""
                <tr>
                    <td>{param}</td>
                    <td>{first_order:.3f if isinstance(first_order, float) else first_order}</td>
                    <td>{total_order if total_order else 'N/A'}</td>
                </tr>
                """
            
            html += """
                </tbody>
            </table>
            """
        
        html += "</div>"
        
        return html
    
    def _add_correlation_section(self, correlation_analysis: Dict) -> str:
        """Add correlation analysis section."""
        html = """
        <section class="analysis">
            <h2>Correlation Analysis</h2>
        """
        
        # Pearson correlations
        if 'pearson' in correlation_analysis:
            pearson = correlation_analysis['pearson']
            if 'significant' in pearson and pearson['significant']:
                html += f"""
                <div class="subsection">
                    <h3>Significant Correlations (p < 0.05)</h3>
                    <p>{', '.join(pearson['significant'])}</p>
                </div>
                """
        
        # PCA results
        if 'pca' in correlation_analysis:
            pca = correlation_analysis['pca']
            html += f"""
            <div class="subsection">
                <h3>Principal Component Analysis</h3>
                <p>Components for 95% variance: {pca['n_components_95_variance']}</p>
                <p>First 3 components explain: {sum(pca['explained_variance_ratio'][:3])*100:.1f}% of variance</p>
            </div>
            """
        
        # Clustering
        if 'clustering' in correlation_analysis:
            clustering = correlation_analysis['clustering']
            html += """
            <div class="subsection">
                <h3>Parameter Clustering</h3>
            """
            
            for cluster_name, params in clustering['clusters'].items():
                html += f"<p><strong>{cluster_name}:</strong> {', '.join(params)}</p>"
            
            html += "</div>"
        
        # Multicollinearity
        if 'multicollinearity' in correlation_analysis:
            multi = correlation_analysis['multicollinearity']
            if multi.get('high_collinearity'):
                html += f"""
                <div class="subsection">
                    <h3>Multicollinearity Warning</h3>
                    <p>High collinearity detected (VIF > 10): {', '.join(multi['high_collinearity'])}</p>
                </div>
                """
        
        html += "</section>"
        
        return html
    
    def _add_visualizations_section(self, identifier: str) -> str:
        """Add visualizations section with embedded images."""
        html = """
        <section class="visualizations">
            <h2>Visualizations</h2>
        """
        
        # Find relevant plot files
        plot_patterns = {
            'single': ['single_param_*.png', 'quick_analysis_*.png'],
            'multi_parameter': ['few_params_*.png', 'parameter_ranking_*.png', 
                              'interaction_heatmap_*.png'],
            'comprehensive': ['tornado_plot_*.png', 'correlation_heatmap_*.png',
                            'pca_analysis_*.png', 'parameter_dendrogram_*.png']
        }
        
        # For single parameter, use the parameter name as identifier
        if identifier not in plot_patterns:
            # Assume it's a parameter name for single analysis
            patterns = [f'single_param_{identifier}_*.png', f'{identifier}_*.png']
        else:
            patterns = plot_patterns.get(identifier, [])
        
        # Also always include summary plot
        patterns.append('summary_plot_*.png')
        
        plots_found = []
        for pattern in patterns:
            plots_found.extend(list(self.figure_dir.glob(pattern)))
        
        # Sort by modification time (newest first)
        plots_found.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        if plots_found:
            for plot_path in plots_found[:6]:  # Limit to 6 most recent plots
                # Try to embed image as base64
                try:
                    with open(plot_path, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode()
                    
                    html += f"""
                    <div class="visualization">
                        <h3>{plot_path.stem.replace('_', ' ').title()}</h3>
                        <img src="data:image/png;base64,{img_data}" alt="{plot_path.stem}">
                    </div>
                    """
                except:
                    # Fallback to file path reference
                    html += f"""
                    <div class="visualization">
                        <h3>{plot_path.stem.replace('_', ' ').title()}</h3>
                        <p>Image: {plot_path}</p>
                    </div>
                    """
        else:
            html += "<p>No visualizations available.</p>"
        
        html += "</section>"
        
        return html
    
    def _create_recommendations_section(self, recommendations: List[Dict]) -> str:
        """Create recommendations section."""
        html = """
        <section class="recommendations">
            <h2>Recommendations</h2>
        """
        
        if not recommendations:
            html += "<p>No specific recommendations generated.</p>"
        else:
            # Group by priority
            high = [r for r in recommendations if r['priority'] == 'high']
            medium = [r for r in recommendations if r['priority'] == 'medium']
            low = [r for r in recommendations if r['priority'] == 'low']
            
            if high:
                html += """
                <div class="subsection">
                    <h3 class="priority-high">High Priority</h3>
                    <ul>
                """
                for r in high:
                    html += f"<li>{r['message']}</li>"
                html += "</ul></div>"
            
            if medium:
                html += """
                <div class="subsection">
                    <h3 class="priority-medium">Medium Priority</h3>
                    <ul>
                """
                for r in medium:
                    html += f"<li>{r['message']}</li>"
                html += "</ul></div>"
            
            if low:
                html += """
                <div class="subsection">
                    <h3 class="priority-low">Low Priority</h3>
                    <ul>
                """
                for r in low:
                    html += f"<li>{r['message']}</li>"
                html += "</ul></div>"
        
        html += "</section>"
        
        return html
    
    def _create_summary_statistics_section(self, summary_stats: Dict) -> str:
        """Create summary statistics section."""
        html = """
        <section class="analysis">
            <h2>Summary Statistics</h2>
        """
        
        # Simulation statistics
        html += f"""
        <div class="subsection">
            <h3>Simulation Overview</h3>
            <p>Total simulations: {summary_stats.get('total_simulations', 0)}</p>
            <p>Successful simulations: {summary_stats.get('successful_simulations', 0)}</p>
        </div>
        """
        
        # Metric statistics
        if 'metric_statistics' in summary_stats:
            m_stats = summary_stats['metric_statistics']
            html += f"""
            <div class="subsection">
                <h3>Target Metric Statistics</h3>
                <table>
                    <tr><td>Mean</td><td>{m_stats['mean']:.2f}</td></tr>
                    <tr><td>Std Dev</td><td>{m_stats['std']:.2f}</td></tr>
                    <tr><td>Minimum</td><td>{m_stats['min']:.2f}</td></tr>
                    <tr><td>Q25</td><td>{m_stats['q25']:.2f}</td></tr>
                    <tr><td>Median</td><td>{m_stats['median']:.2f}</td></tr>
                    <tr><td>Q75</td><td>{m_stats['q75']:.2f}</td></tr>
                    <tr><td>Maximum</td><td>{m_stats['max']:.2f}</td></tr>
                </table>
            </div>
            """
        
        # Execution statistics
        if 'execution_statistics' in summary_stats:
            e_stats = summary_stats['execution_statistics']
            html += f"""
            <div class="subsection">
                <h3>Execution Statistics</h3>
                <p>Total execution time: {e_stats['total_time']:.1f} seconds</p>
                <p>Mean execution time: {e_stats['mean_time']:.1f} seconds</p>
                <p>Max execution time: {e_stats['max_time']:.1f} seconds</p>
            </div>
            """
        
        html += "</section>"
        
        return html
    
    def _create_data_table(self, df: pd.DataFrame, parameters: List[str], 
                           metric: str, max_rows: int = 50) -> str:
        """Create data table section."""
        html = """
        <section class="analysis">
            <h2>Simulation Data</h2>
            <div class="data-table">
        """
        
        # Select columns to display
        columns = parameters + [metric]
        if 'execution_time' in df.columns:
            columns.append('execution_time')
        
        # Filter and sort dataframe
        df_display = df[columns].head(max_rows)
        
        # Convert to HTML table
        html += df_display.to_html(index=False, classes='data-table')
        
        if len(df) > max_rows:
            html += f"<p><em>Showing first {max_rows} of {len(df)} simulations</em></p>"
        
        html += """
            </div>
        </section>
        """
        
        return html


def main():
    """Main function for testing the report generator."""
    from results_analyzer import ResultsAnalyzer
    
    # Load and analyze results
    analyzer = ResultsAnalyzer()
    df = analyzer.load_results()
    
    if not df.empty:
        analysis_results = analyzer.analyze()
        
        # Add some mock components for testing
        analysis_results['components'] = {
            'core_analysis': analysis_results,
            'summary_statistics': {
                'total_simulations': len(df),
                'successful_simulations': len(df),
                'metric_statistics': {
                    'mean': df['trap_count'].mean() if 'trap_count' in df.columns else 0,
                    'std': df['trap_count'].std() if 'trap_count' in df.columns else 0,
                    'min': df['trap_count'].min() if 'trap_count' in df.columns else 0,
                    'max': df['trap_count'].max() if 'trap_count' in df.columns else 0,
                    'median': df['trap_count'].median() if 'trap_count' in df.columns else 0,
                    'q25': df['trap_count'].quantile(0.25) if 'trap_count' in df.columns else 0,
                    'q75': df['trap_count'].quantile(0.75) if 'trap_count' in df.columns else 0
                }
            }
        }
        
        # Generate report
        generator = ReportGenerator()
        report_path = generator.generate_report(analysis_results, df)
        
        print(f"Report generated: {report_path}")
    else:
        print("No data available for report generation")


if __name__ == "__main__":
    main()