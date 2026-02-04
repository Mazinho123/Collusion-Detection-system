"""
Visualization Module
Creates visualizations for collusion detection results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import networkx as nx
from pathlib import Path


class CollusionVisualizer:
    """Generate visualizations for collusion analysis"""
    
    def __init__(self, output_dir: str = './results', dpi: int = 100):
        """Initialize visualizer"""
        self.output_dir = output_dir
        self.dpi = dpi
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        sns.set_style("darkgrid")
    
    def plot_risk_distribution(self, risk_scores_df: pd.DataFrame) -> None:
        """Plot distribution of risk scores"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(risk_scores_df['combined_risk_score'], bins=30, color='skyblue', edgecolor='black')
        axes[0].axvline(risk_scores_df['combined_risk_score'].mean(), color='red', linestyle='--', label='Mean')
        axes[0].set_xlabel('Risk Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Collusion Risk Scores')
        axes[0].legend()
        
        # Risk level pie chart
        risk_counts = risk_scores_df['risk_level'].value_counts()
        colors = {'HIGH': '#FF6B6B', 'MEDIUM': '#FFA500', 'LOW': '#FFD700', 'VERY_LOW': '#90EE90'}
        color_list = [colors.get(level, '#808080') for level in risk_counts.index]
        axes[1].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', colors=color_list)
        axes[1].set_title('Distribution by Risk Level')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/risk_distribution.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_score_components(self, risk_scores_df: pd.DataFrame) -> None:
        """Plot components of risk scores"""
        components = ['anomaly_score', 'network_score', 'behavior_score']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(risk_scores_df.head(20)))
        width = 0.25
        
        for i, component in enumerate(components):
            ax.bar(x + i*width, risk_scores_df[component].head(20), width, label=component)
        
        ax.set_xlabel('Bidder ID')
        ax.set_ylabel('Score')
        ax.set_title('Risk Score Components (Top 20 Bidders)')
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"B{bid}" for bid in risk_scores_df['bidder_id'].head(20)])
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/score_components.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_network_graph(self, network: nx.Graph, top_n: int = 30) -> None:
        """Plot bidder network"""
        # Get top nodes by degree
        degrees = dict(network.degree(weight='weight'))
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_node_ids = [node for node, _ in top_nodes]
        
        # Create subgraph
        subgraph = network.subgraph(top_node_ids)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Layout
        pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
        
        # Draw edges
        edges = subgraph.edges()
        weights = [subgraph[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(subgraph, pos, width=[w*0.5 for w in weights], alpha=0.5, ax=ax)
        
        # Draw nodes
        node_sizes = [degrees.get(node, 1) * 100 for node in subgraph.nodes()]
        nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, node_color='lightblue', ax=ax)
        
        # Draw labels
        labels = {node: f"B{node}" for node in subgraph.nodes()}
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8, ax=ax)
        
        ax.set_title(f'Bidder Co-bidding Network (Top {top_n} Nodes)')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/network_graph.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_anomaly_scores(self, anomaly_scores_df: pd.DataFrame) -> None:
        """Plot anomaly scores"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sorted_df = anomaly_scores_df.sort_values('ensemble_score', ascending=False).head(30)
        
        colors = ['red' if x else 'blue' for x in sorted_df['is_anomaly']]
        ax.barh(range(len(sorted_df)), sorted_df['ensemble_score'], color=colors)
        
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels([f"B{bid}" for bid in sorted_df['bidder_id']])
        ax.set_xlabel('Ensemble Anomaly Score')
        ax.set_title('Top 30 Bidders by Anomaly Score')
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/anomaly_scores.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self, importance_dict: Dict[str, float]) -> None:
        """Plot feature importance"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        features = list(importance_dict.keys())
        values = list(importance_dict.values())
        
        ax.barh(features, values, color='steelblue')
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance for Anomaly Detection')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_importance.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_bidding_timeline(self, bids_df: pd.DataFrame, top_bidders: int = 5) -> None:
        """Plot bidding timeline"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        top_bidder_ids = bids_df.groupby('bidder_id').size().nlargest(top_bidders).index
        
        for bidder_id in top_bidder_ids:
            bidder_bids = bids_df[bids_df['bidder_id'] == bidder_id].sort_values('bid_time')
            ax.plot(range(len(bidder_bids)), bidder_bids['bid_price'], marker='o', label=f'Bidder {bidder_id}')
        
        ax.set_xlabel('Bid Number')
        ax.set_ylabel('Bid Price')
        ax.set_title('Bidding Timeline for Top Bidders')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/bidding_timeline.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_auction_competition(self, bids_df: pd.DataFrame, auctions_df: pd.DataFrame) -> None:
        """Plot auction competition levels"""
        bids_per_auction = bids_df.groupby('auction_id').size()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(bids_per_auction, bins=20, color='green', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Number of Bids per Auction')
        ax.set_ylabel('Frequency')
        ax.set_title('Auction Competition Level Distribution')
        ax.axvline(bids_per_auction.mean(), color='red', linestyle='--', label=f'Mean: {bids_per_auction.mean():.1f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/auction_competition.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def generate_all_visualizations(self, risk_scores_df: pd.DataFrame, 
                                   anomaly_scores_df: pd.DataFrame,
                                   bids_df: pd.DataFrame,
                                   auctions_df: pd.DataFrame,
                                   network: nx.Graph,
                                   importance_dict: Dict[str, float]) -> None:
        """Generate all visualizations"""
        print("Generating visualizations...")
        
        self.plot_risk_distribution(risk_scores_df)
        print("✓ Risk distribution")
        
        self.plot_score_components(risk_scores_df)
        print("✓ Score components")
        
        self.plot_network_graph(network)
        print("✓ Network graph")
        
        self.plot_anomaly_scores(anomaly_scores_df)
        print("✓ Anomaly scores")
        
        self.plot_feature_importance(importance_dict)
        print("✓ Feature importance")
        
        self.plot_bidding_timeline(bids_df)
        print("✓ Bidding timeline")
        
        self.plot_auction_competition(bids_df, auctions_df)
        print("✓ Auction competition")
        
        print(f"\nAll visualizations saved to {self.output_dir}")
