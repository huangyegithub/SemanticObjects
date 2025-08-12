#!/usr/bin/env python3
"""
Correlation Analysis Module for SMOL Sensitivity Analysis

This module provides statistical correlation analysis capabilities including
Pearson/Spearman correlations, PCA, and parameter clustering.

Author: SMOL Sensitivity Analysis Framework
Date: 2025
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class CorrelationAnalyzer:
    """
    Statistical correlation analyzer for sensitivity analysis results.
    
    Provides correlation analysis, PCA, and clustering for parameters.
    """
    
    def __init__(self, figure_dir: str = "../figures"):
        """
        Initialize the correlation analyzer.
        
        Args:
            figure_dir: Directory to save figures
        """
        self.figure_dir = Path(figure_dir)
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        
    def analyze_correlations(self, results_df: pd.DataFrame, 
                            parameters: List[str],
                            target_metric: str = "trap_count") -> Dict[str, Any]:
        """
        Perform comprehensive correlation analysis.
        
        Args:
            results_df: DataFrame with simulation results
            parameters: List of parameter names
            target_metric: Target metric to analyze
            
        Returns:
            Dictionary containing correlation analysis results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'n_parameters': len(parameters),
            'n_simulations': len(results_df),
            'target_metric': target_metric
        }
        
        # Filter to valid parameters and metric
        valid_cols = [p for p in parameters if p in results_df.columns] + [target_metric]
        df_analysis = results_df[valid_cols].dropna()
        
        if len(df_analysis) < 3:
            results['error'] = 'Insufficient data for correlation analysis'
            return results
        
        # Calculate correlation matrices
        results['pearson'] = self._calculate_pearson_correlation(df_analysis, parameters, target_metric)
        results['spearman'] = self._calculate_spearman_correlation(df_analysis, parameters, target_metric)
        
        # Perform PCA if enough parameters
        if len(parameters) > 2:
            results['pca'] = self._perform_pca(df_analysis, parameters, target_metric)
        
        # Cluster parameters if enough
        if len(parameters) > 3:
            results['clustering'] = self._cluster_parameters(df_analysis, parameters)
        
        # Calculate partial correlations
        if len(parameters) > 1:
            results['partial_correlations'] = self._calculate_partial_correlations(
                df_analysis, parameters, target_metric
            )
        
        # Identify multicollinearity
        if len(parameters) > 1:
            results['multicollinearity'] = self._detect_multicollinearity(df_analysis, parameters)
        
        return results
    
    def _calculate_pearson_correlation(self, df: pd.DataFrame, 
                                      parameters: List[str], 
                                      target_metric: str) -> Dict:
        """Calculate Pearson correlation coefficients."""
        results = {
            'correlations': {},
            'p_values': {},
            'significant': []
        }
        
        for param in parameters:
            if param in df.columns:
                corr, p_value = stats.pearsonr(df[param], df[target_metric])
                results['correlations'][param] = float(corr)
                results['p_values'][param] = float(p_value)
                
                if p_value < 0.05:
                    results['significant'].append(param)
        
        # Full correlation matrix
        corr_matrix = df[parameters + [target_metric]].corr(method='pearson')
        results['matrix'] = corr_matrix.to_dict()
        
        return results
    
    def _calculate_spearman_correlation(self, df: pd.DataFrame, 
                                       parameters: List[str], 
                                       target_metric: str) -> Dict:
        """Calculate Spearman rank correlation coefficients."""
        results = {
            'correlations': {},
            'p_values': {},
            'monotonic': []
        }
        
        for param in parameters:
            if param in df.columns:
                corr, p_value = stats.spearmanr(df[param], df[target_metric])
                results['correlations'][param] = float(corr)
                results['p_values'][param] = float(p_value)
                
                # Check for monotonic relationship
                if abs(corr) > 0.8:
                    results['monotonic'].append(param)
        
        # Full correlation matrix
        corr_matrix = df[parameters + [target_metric]].corr(method='spearman')
        results['matrix'] = corr_matrix.to_dict()
        
        return results
    
    def _perform_pca(self, df: pd.DataFrame, 
                    parameters: List[str], 
                    target_metric: str) -> Dict:
        """Perform Principal Component Analysis."""
        # Prepare data
        X = df[parameters].values
        y = df[target_metric].values
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Calculate number of components for 95% variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components_95 = np.argmax(cumsum >= 0.95) + 1
        
        results = {
            'n_components': len(parameters),
            'n_components_95_variance': int(n_components_95),
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': cumsum.tolist(),
            'components': {}
        }
        
        # Store loadings for first few components
        for i in range(min(3, len(parameters))):
            loadings = pca.components_[i]
            results['components'][f'PC{i+1}'] = {
                'loadings': dict(zip(parameters, loadings.tolist())),
                'variance_explained': float(pca.explained_variance_ratio_[i])
            }
        
        # Calculate correlation between PCs and target metric
        pc_correlations = {}
        for i in range(min(3, X_pca.shape[1])):
            corr, p_value = stats.pearsonr(X_pca[:, i], y)
            pc_correlations[f'PC{i+1}'] = {
                'correlation': float(corr),
                'p_value': float(p_value)
            }
        results['pc_target_correlations'] = pc_correlations
        
        # Create PCA visualizations
        self._plot_pca_results(pca, parameters, X_pca, y)
        
        return results
    
    def _cluster_parameters(self, df: pd.DataFrame, parameters: List[str]) -> Dict:
        """Perform hierarchical clustering of parameters."""
        # Calculate correlation matrix
        corr_matrix = df[parameters].corr()
        
        # Convert correlation to distance
        distance_matrix = 1 - abs(corr_matrix)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(distance_matrix, method='ward')
        
        # Determine optimal number of clusters using elbow method
        max_clusters = min(len(parameters) - 1, 5)
        inertias = []
        
        for k in range(2, max_clusters + 1):
            clusters = fcluster(linkage_matrix, k, criterion='maxclust')
            # Calculate within-cluster sum of squares
            wcss = 0
            for i in range(1, k + 1):
                cluster_params = [p for j, p in enumerate(parameters) if clusters[j] == i]
                if len(cluster_params) > 1:
                    cluster_corr = corr_matrix.loc[cluster_params, cluster_params]
                    wcss += np.sum((1 - cluster_corr.values) ** 2)
            inertias.append(wcss)
        
        # Find elbow point (simplified)
        if len(inertias) > 1:
            optimal_k = 2  # Default
            if len(inertias) > 2:
                # Find point with maximum second derivative
                second_derivative = np.diff(np.diff(inertias))
                if len(second_derivative) > 0:
                    optimal_k = np.argmax(second_derivative) + 2
        else:
            optimal_k = 2
        
        # Get cluster assignments
        clusters = fcluster(linkage_matrix, optimal_k, criterion='maxclust')
        
        # Organize results
        cluster_dict = {}
        for i in range(1, optimal_k + 1):
            cluster_params = [p for j, p in enumerate(parameters) if clusters[j] == i]
            cluster_dict[f'Cluster_{i}'] = cluster_params
        
        results = {
            'optimal_clusters': int(optimal_k),
            'clusters': cluster_dict,
            'linkage_matrix': linkage_matrix.tolist()
        }
        
        # Create dendrogram
        self._plot_dendrogram(linkage_matrix, parameters)
        
        return results
    
    def _calculate_partial_correlations(self, df: pd.DataFrame, 
                                       parameters: List[str], 
                                       target_metric: str) -> Dict:
        """Calculate partial correlations controlling for other parameters."""
        results = {}
        
        for param in parameters:
            # Control variables (all other parameters)
            control_vars = [p for p in parameters if p != param]
            
            if len(control_vars) > 0:
                # Calculate partial correlation
                partial_corr = self._partial_correlation(
                    df, param, target_metric, control_vars
                )
                results[param] = {
                    'partial_correlation': float(partial_corr),
                    'controlled_for': control_vars
                }
        
        return results
    
    def _partial_correlation(self, df: pd.DataFrame, x: str, y: str, 
                            control: List[str]) -> float:
        """
        Calculate partial correlation between x and y controlling for control variables.
        """
        from sklearn.linear_model import LinearRegression
        
        # Residualize x
        model_x = LinearRegression()
        model_x.fit(df[control], df[x])
        residuals_x = df[x] - model_x.predict(df[control])
        
        # Residualize y
        model_y = LinearRegression()
        model_y.fit(df[control], df[y])
        residuals_y = df[y] - model_y.predict(df[control])
        
        # Calculate correlation of residuals
        corr, _ = stats.pearsonr(residuals_x, residuals_y)
        
        return corr
    
    def _detect_multicollinearity(self, df: pd.DataFrame, parameters: List[str]) -> Dict:
        """Detect multicollinearity using VIF (Variance Inflation Factor)."""
        from sklearn.linear_model import LinearRegression
        
        results = {
            'vif_scores': {},
            'high_collinearity': []
        }
        
        for i, param in enumerate(parameters):
            # Use other parameters as predictors
            other_params = [p for j, p in enumerate(parameters) if j != i]
            
            if len(other_params) > 0:
                # Fit regression
                model = LinearRegression()
                model.fit(df[other_params], df[param])
                r_squared = model.score(df[other_params], df[param])
                
                # Calculate VIF
                if r_squared < 1:
                    vif = 1 / (1 - r_squared)
                else:
                    vif = np.inf
                
                results['vif_scores'][param] = float(vif)
                
                # Flag high collinearity (VIF > 10)
                if vif > 10:
                    results['high_collinearity'].append(param)
        
        return results
    
    def _plot_pca_results(self, pca, parameters: List[str], 
                         X_pca: np.ndarray, y: np.ndarray):
        """Create PCA visualization plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Scree plot
        ax = axes[0, 0]
        ax.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
               pca.explained_variance_ratio_)
        ax.plot(range(1, len(pca.explained_variance_ratio_) + 1),
               np.cumsum(pca.explained_variance_ratio_), 'r-o')
        ax.axhline(y=0.95, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Variance Explained')
        ax.set_title('PCA Scree Plot')
        ax.legend(['Cumulative', '95% threshold', 'Individual'])
        ax.grid(True, alpha=0.3)
        
        # 2. Biplot (PC1 vs PC2)
        if X_pca.shape[1] >= 2:
            ax = axes[0, 1]
            
            # Plot samples
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, 
                               cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, ax=ax, label='Target Metric')
            
            # Plot loadings
            for i, param in enumerate(parameters):
                ax.arrow(0, 0, 
                        pca.components_[0, i] * 3,
                        pca.components_[1, i] * 3,
                        color='r', alpha=0.5, head_width=0.05)
                ax.text(pca.components_[0, i] * 3.2,
                       pca.components_[1, i] * 3.2,
                       param, fontsize=9)
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.set_title('PCA Biplot')
            ax.grid(True, alpha=0.3)
        
        # 3. Loadings heatmap
        ax = axes[1, 0]
        n_components = min(5, len(parameters))
        loadings = pca.components_[:n_components, :]
        
        sns.heatmap(loadings, xticklabels=parameters,
                   yticklabels=[f'PC{i+1}' for i in range(n_components)],
                   cmap='coolwarm', center=0, annot=True, fmt='.2f',
                   ax=ax, cbar_kws={'label': 'Loading'})
        ax.set_title('PCA Loadings')
        
        # 4. Correlation with target
        if X_pca.shape[1] >= 3:
            ax = axes[1, 1]
            pc_correlations = []
            for i in range(min(5, X_pca.shape[1])):
                corr, _ = stats.pearsonr(X_pca[:, i], y)
                pc_correlations.append(corr)
            
            ax.bar(range(1, len(pc_correlations) + 1), pc_correlations)
            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Correlation with Target Metric')
            ax.set_title('PC Correlation with Target')
            ax.grid(True, alpha=0.3)
        else:
            axes[1, 1].axis('off')
        
        plt.suptitle('PCA Analysis Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.figure_dir / f"pca_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
    
    def _plot_dendrogram(self, linkage_matrix: np.ndarray, parameters: List[str]):
        """Plot hierarchical clustering dendrogram."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        dendrogram(linkage_matrix, labels=parameters, ax=ax,
                  orientation='right', distance_sort='descending')
        
        ax.set_xlabel('Distance (1 - |correlation|)')
        ax.set_title('Hierarchical Clustering of Parameters', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.figure_dir / f"parameter_dendrogram_{timestamp}.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
    
    def create_correlation_report(self, correlation_results: Dict) -> str:
        """
        Create a text report of correlation analysis results.
        
        Args:
            correlation_results: Results from analyze_correlations
            
        Returns:
            Formatted text report
        """
        report = []
        report.append("="*60)
        report.append("CORRELATION ANALYSIS REPORT")
        report.append("="*60)
        report.append(f"Generated: {correlation_results['timestamp']}")
        report.append(f"Parameters: {correlation_results['n_parameters']}")
        report.append(f"Simulations: {correlation_results['n_simulations']}")
        report.append(f"Target Metric: {correlation_results['target_metric']}")
        report.append("")
        
        # Pearson correlations
        if 'pearson' in correlation_results:
            report.append("-"*40)
            report.append("PEARSON CORRELATIONS")
            report.append("-"*40)
            
            pearson = correlation_results['pearson']
            corr_items = sorted(pearson['correlations'].items(), 
                              key=lambda x: abs(x[1]), reverse=True)
            
            for param, corr in corr_items:
                p_val = pearson['p_values'][param]
                sig = "*" if p_val < 0.05 else " "
                report.append(f"{param:20s}: {corr:7.3f} (p={p_val:.3e}) {sig}")
            
            if pearson['significant']:
                report.append(f"\nSignificant (p<0.05): {', '.join(pearson['significant'])}")
        
        # Spearman correlations
        if 'spearman' in correlation_results:
            report.append("")
            report.append("-"*40)
            report.append("SPEARMAN CORRELATIONS")
            report.append("-"*40)
            
            spearman = correlation_results['spearman']
            if spearman['monotonic']:
                report.append(f"Monotonic (|r|>0.8): {', '.join(spearman['monotonic'])}")
        
        # PCA results
        if 'pca' in correlation_results:
            report.append("")
            report.append("-"*40)
            report.append("PRINCIPAL COMPONENT ANALYSIS")
            report.append("-"*40)
            
            pca = correlation_results['pca']
            report.append(f"Components for 95% variance: {pca['n_components_95_variance']}")
            report.append("\nVariance explained by components:")
            
            for i, var in enumerate(pca['explained_variance_ratio'][:5]):
                cum_var = pca['cumulative_variance'][i]
                report.append(f"  PC{i+1}: {var*100:5.1f}% (cumulative: {cum_var*100:5.1f}%)")
        
        # Clustering results
        if 'clustering' in correlation_results:
            report.append("")
            report.append("-"*40)
            report.append("PARAMETER CLUSTERING")
            report.append("-"*40)
            
            clustering = correlation_results['clustering']
            report.append(f"Optimal clusters: {clustering['optimal_clusters']}")
            
            for cluster_name, params in clustering['clusters'].items():
                report.append(f"\n{cluster_name}:")
                for param in params:
                    report.append(f"  - {param}")
        
        # Multicollinearity
        if 'multicollinearity' in correlation_results:
            report.append("")
            report.append("-"*40)
            report.append("MULTICOLLINEARITY (VIF)")
            report.append("-"*40)
            
            multi = correlation_results['multicollinearity']
            if multi['high_collinearity']:
                report.append(f"High collinearity (VIF>10): {', '.join(multi['high_collinearity'])}")
            else:
                report.append("No high collinearity detected")
        
        return "\n".join(report)


def main():
    """Main function for testing the correlation analyzer."""
    from results_analyzer import ResultsAnalyzer
    
    # Load results
    analyzer = ResultsAnalyzer()
    df = analyzer.load_results()
    
    # Perform correlation analysis
    corr_analyzer = CorrelationAnalyzer()
    
    if analyzer.varied_parameters:
        correlation_results = corr_analyzer.analyze_correlations(
            df, analyzer.varied_parameters, "trap_count"
        )
        
        # Generate report
        report = corr_analyzer.create_correlation_report(correlation_results)
        print(report)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"../data/output/correlation_analysis_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(correlation_results, f, indent=2)
        
        print(f"\nCorrelation analysis saved to {output_file}")
    else:
        print("No varied parameters found for correlation analysis")


if __name__ == "__main__":
    main()