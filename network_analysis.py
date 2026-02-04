"""
Network Analysis Module
Analyzes bidder networks and detects collusive groups
"""

import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any


class BidderNetworkAnalysis:
    """Analyze bidder co-bidding networks to detect cartels"""
    
    def __init__(self, bids_df: pd.DataFrame, min_coprediction_threshold: int = 3):
        """Initialize network analyzer"""
        self.bids_df = bids_df.copy()
        self.min_coprediction_threshold = min_coprediction_threshold
        self.network = None
        self.communities = None
        self.suspicious_pairs = None
    
    def build_cobidding_network(self) -> nx.Graph:
        """Build network of bidders based on co-bidding"""
        G = nx.Graph()
        
        # Get all bidders
        all_bidders = self.bids_df['bidder_id'].unique()
        for bidder in all_bidders:
            G.add_node(bidder)
        
        # Add edges for co-bidders
        auction_bidders = self.bids_df.groupby('auction_id')['bidder_id'].apply(list)
        edge_counts = defaultdict(int)
        
        for bidders_in_auction in auction_bidders:
            # Create edges for all pairs
            for i in range(len(bidders_in_auction)):
                for j in range(i + 1, len(bidders_in_auction)):
                    bidder1 = min(bidders_in_auction[i], bidders_in_auction[j])
                    bidder2 = max(bidders_in_auction[i], bidders_in_auction[j])
                    edge_counts[(bidder1, bidder2)] += 1
        
        # Add edges with threshold
        for (bidder1, bidder2), count in edge_counts.items():
            if count >= self.min_coprediction_threshold:
                G.add_edge(bidder1, bidder2, weight=count)
        
        self.network = G
        return G
    
    def detect_communities(self) -> Dict[int, List[int]]:
        """Detect communities using Louvain method"""
        if self.network is None:
            self.build_cobidding_network()
        
        try:
            import community as community_louvain
            communities_dict = community_louvain.best_partition(self.network)
        except ImportError:
            # Fallback: use connected components
            components = list(nx.connected_components(self.network))
            communities_dict = {}
            for comp_id, component in enumerate(components):
                for node in component:
                    communities_dict[node] = comp_id
        
        # Reorganize by community ID
        communities = defaultdict(list)
        for node, comm_id in communities_dict.items():
            communities[comm_id].append(node)
        
        self.communities = dict(communities)
        return self.communities
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get network statistics"""
        if self.network is None:
            self.build_cobidding_network()
        
        stats = {
            'num_nodes': self.network.number_of_nodes(),
            'num_edges': self.network.number_of_edges(),
            'density': nx.density(self.network),
            'num_components': nx.number_connected_components(self.network),
            'average_clustering': nx.average_clustering(self.network) if self.network.number_of_nodes() > 0 else 0
        }
        
        return stats
    
    def get_bidder_centrality(self) -> Dict[int, Dict[str, float]]:
        """Calculate centrality measures for each bidder"""
        if self.network is None:
            self.build_cobidding_network()
        
        degree_centrality = nx.degree_centrality(self.network)
        betweenness_centrality = nx.betweenness_centrality(self.network, weight='weight')
        closeness_centrality = nx.closeness_centrality(self.network)
        
        centrality_measures = {}
        for bidder in self.network.nodes():
            centrality_measures[bidder] = {
                'degree_centrality': degree_centrality.get(bidder, 0),
                'betweenness_centrality': betweenness_centrality.get(bidder, 0),
                'closeness_centrality': closeness_centrality.get(bidder, 0)
            }
        
        return centrality_measures
    
    def find_suspicious_pairs(self) -> List[Tuple[int, int, int]]:
        """Find pairs of bidders that frequently co-bid"""
        if self.network is None:
            self.build_cobidding_network()
        
        suspicious_pairs = []
        for edge in self.network.edges(data=True):
            bidder1, bidder2, data = edge
            weight = data.get('weight', 0)
            suspicious_pairs.append((bidder1, bidder2, weight))
        
        # Sort by weight
        suspicious_pairs.sort(key=lambda x: x[2], reverse=True)
        self.suspicious_pairs = suspicious_pairs
        
        return suspicious_pairs
    
    def get_bidder_network_score(self, bidder_id: int) -> Dict[str, float]:
        """Calculate network-based score for a bidder"""
        if self.network is None:
            self.build_cobidding_network()
        
        if bidder_id not in self.network.nodes():
            return {
                'network_degree': 0,
                'network_clustering': 0,
                'community_size': 0,
                'network_score': 0
            }
        
        # Degree (how many bidders co-bid with this bidder)
        degree = self.network.degree(bidder_id)
        max_degree = max(dict(self.network.degree()).values()) if len(self.network) > 0 else 1
        network_degree = degree / max_degree if max_degree > 0 else 0
        
        # Local clustering coefficient
        clustering = nx.clustering(self.network, bidder_id)
        
        # Community size
        if self.communities is None:
            self.detect_communities()
        
        community_id = None
        community_size = 0
        for comm_id, members in self.communities.items():
            if bidder_id in members:
                community_id = comm_id
                community_size = len(members)
                break
        
        # Normalized community size
        total_bidders = self.network.number_of_nodes()
        normalized_community_size = community_size / total_bidders if total_bidders > 0 else 0
        
        # Network score (weighted combination)
        network_score = 0.5 * network_degree + 0.3 * clustering + 0.2 * normalized_community_size
        
        return {
            'network_degree': network_degree,
            'network_clustering': clustering,
            'community_size': community_size,
            'network_score': network_score
        }
    
    def get_network_analysis_summary(self) -> Dict[str, Any]:
        """Get comprehensive network analysis summary"""
        if self.network is None:
            self.build_cobidding_network()
        
        if self.communities is None:
            self.detect_communities()
        
        return {
            'network_statistics': self.get_network_statistics(),
            'num_communities': len(self.communities),
            'largest_community_size': max([len(c) for c in self.communities.values()]) if self.communities else 0,
            'num_suspicious_pairs': len(self.find_suspicious_pairs()),
            'top_suspicious_pairs': self.find_suspicious_pairs()[:5]
        }
