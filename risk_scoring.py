"""
Risk Scoring Module
Combines detection signals into comprehensive collusion risk scores
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple


class CollusionRiskScorer:
    """Combine detection signals into collusion risk scores"""
    
    def __init__(self, features_df: pd.DataFrame, anomaly_scores_df: pd.DataFrame, 
                 network_scores: Dict[int, Dict[str, float]], config: Dict[str, float] = None):
        """Initialize risk scorer"""
        self.features_df = features_df.copy()
        self.anomaly_scores_df = anomaly_scores_df.copy()
        self.network_scores = network_scores
        
        # Default weights
        self.anomaly_weight = 0.5
        self.network_weight = 0.3
        self.behavior_weight = 0.2
        
        if config:
            self.anomaly_weight = config.get('anomaly_score_weight', 0.5)
            self.network_weight = config.get('network_score_weight', 0.3)
            self.behavior_weight = config.get('behavior_score_weight', 0.2)
        
        # Risk thresholds
        self.low_threshold = 0.3
        self.medium_threshold = 0.6
        self.high_threshold = 0.8
        
        if config:
            self.low_threshold = config.get('risk_threshold_low', 0.3)
            self.medium_threshold = config.get('risk_threshold_medium', 0.6)
            self.high_threshold = config.get('risk_threshold_high', 0.8)
        
        self.risk_scores = None
    
    def calculate_behavior_score(self, bidder_id: int) -> float:
        """Calculate behavior-based risk score"""
        bidder_data = self.features_df[self.features_df['bidder_id'] == bidder_id]
        
        if len(bidder_data) == 0:
            return 0.0
        
        row = bidder_data.iloc[0]
        
        # Factors indicating collusion
        factors = []
        
        # High win rate (suspicious)
        if 'win_rate' in row:
            factors.append(row['win_rate'])  # High win rate = higher score
        
        # Low bid position variance (suspicious - always bids early)
        if 'first_bid_ratio' in row:
            factors.append(row['first_bid_ratio'])
        
        # High withdrawal rate (suspicious)
        if 'withdrawal_rate' in row:
            factors.append(row['withdrawal_rate'])
        
        # Low price coefficient of variation (suspicious - consistent prices)
        if 'bid_price_cv' in row:
            cv = row['bid_price_cv']
            factors.append(1 / (1 + cv))  # Low variation = higher score
        
        # High participation rate
        if 'participation_rate' in row:
            factors.append(row['participation_rate'])
        
        behavior_score = np.mean(factors) if factors else 0.0
        return min(behavior_score, 1.0)
    
    def calculate_combined_risk_score(self, bidder_id: int) -> Dict[str, float]:
        """Calculate combined risk score for a bidder"""
        # Get anomaly score
        anomaly_data = self.anomaly_scores_df[self.anomaly_scores_df['bidder_id'] == bidder_id]
        anomaly_score = anomaly_data['anomaly_probability'].values[0] if len(anomaly_data) > 0 else 0.0
        
        # Get network score
        network_score_dict = self.network_scores.get(bidder_id, {})
        network_score = network_score_dict.get('network_score', 0.0)
        
        # Get behavior score
        behavior_score = self.calculate_behavior_score(bidder_id)
        
        # Combine with weights
        combined_score = (
            self.anomaly_weight * anomaly_score +
            self.network_weight * network_score +
            self.behavior_weight * behavior_score
        )
        
        # Normalize
        combined_score = combined_score / (self.anomaly_weight + self.network_weight + self.behavior_weight)
        
        return {
            'bidder_id': bidder_id,
            'anomaly_score': anomaly_score,
            'network_score': network_score,
            'behavior_score': behavior_score,
            'combined_risk_score': combined_score,
            'risk_level': self.get_risk_level(combined_score)
        }
    
    def get_risk_level(self, score: float) -> str:
        """Get risk level from score"""
        if score >= self.high_threshold:
            return 'HIGH'
        elif score >= self.medium_threshold:
            return 'MEDIUM'
        elif score >= self.low_threshold:
            return 'LOW'
        else:
            return 'VERY_LOW'
    
    def calculate_all_risk_scores(self) -> pd.DataFrame:
        """Calculate risk scores for all bidders"""
        risk_scores_list = []
        
        for bidder_id in self.features_df['bidder_id']:
            risk_score = self.calculate_combined_risk_score(bidder_id)
            risk_scores_list.append(risk_score)
        
        self.risk_scores = pd.DataFrame(risk_scores_list)
        return self.risk_scores.sort_values('combined_risk_score', ascending=False)
    
    def get_suspected_colluders(self, min_risk_level: str = 'MEDIUM') -> pd.DataFrame:
        """Get bidders suspected of collusion above risk level"""
        if self.risk_scores is None:
            self.calculate_all_risk_scores()
        
        risk_levels = ['VERY_LOW', 'LOW', 'MEDIUM', 'HIGH']
        min_level_idx = risk_levels.index(min_risk_level)
        
        suspected_mask = self.risk_scores['risk_level'].apply(
            lambda x: risk_levels.index(x) >= min_level_idx
        )
        
        return self.risk_scores[suspected_mask]
    
    def explain_risk_score(self, bidder_id: int) -> Dict[str, Any]:
        """Explain risk score for a bidder"""
        if self.risk_scores is None:
            self.calculate_all_risk_scores()
        
        risk_data = self.risk_scores[self.risk_scores['bidder_id'] == bidder_id]
        
        if len(risk_data) == 0:
            return {}
        
        row = risk_data.iloc[0]
        
        explanation = {
            'bidder_id': bidder_id,
            'combined_risk_score': row['combined_risk_score'],
            'risk_level': row['risk_level'],
            'contributing_factors': []
        }
        
        # Add contributing factors
        if row['anomaly_score'] > 0.6:
            explanation['contributing_factors'].append(
                f"High anomaly score ({row['anomaly_score']:.2f}) - Unusual bidding pattern"
            )
        
        if row['network_score'] > 0.6:
            explanation['contributing_factors'].append(
                f"High network score ({row['network_score']:.2f}) - Frequent co-bidding with suspicious groups"
            )
        
        if row['behavior_score'] > 0.6:
            explanation['contributing_factors'].append(
                f"High behavior score ({row['behavior_score']:.2f}) - Suspicious bidding behavior"
            )
        
        return explanation
    
    def get_scoring_summary(self) -> Dict[str, Any]:
        """Get summary of risk scoring"""
        if self.risk_scores is None:
            self.calculate_all_risk_scores()
        
        return {
            'total_bidders': len(self.risk_scores),
            'high_risk_count': (self.risk_scores['risk_level'] == 'HIGH').sum(),
            'medium_risk_count': (self.risk_scores['risk_level'] == 'MEDIUM').sum(),
            'low_risk_count': (self.risk_scores['risk_level'] == 'LOW').sum(),
            'very_low_risk_count': (self.risk_scores['risk_level'] == 'VERY_LOW').sum(),
            'mean_risk_score': self.risk_scores['combined_risk_score'].mean(),
            'median_risk_score': self.risk_scores['combined_risk_score'].median(),
            'std_risk_score': self.risk_scores['combined_risk_score'].std(),
            'max_risk_score': self.risk_scores['combined_risk_score'].max(),
            'min_risk_score': self.risk_scores['combined_risk_score'].min()
        }
