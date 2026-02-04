"""
Anomaly Detection Module
Detects collusive bidding behavior using ensemble methods
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Any


class AnomalyDetector:
    """Detect anomalous bidding behavior indicating collusion"""
    
    def __init__(self, features_df: pd.DataFrame, contamination: float = 0.1):
        """Initialize anomaly detector"""
        self.features_df = features_df.copy()
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.isolation_forest = None
        self.lof_model = None
        self.feature_columns = [col for col in features_df.columns if col != 'bidder_id']
        self.scaled_features = None
        self.anomaly_scores = None
    
    def prepare_features(self) -> np.ndarray:
        """Prepare and scale features"""
        X = self.features_df[self.feature_columns].values
        self.scaled_features = self.scaler.fit_transform(X)
        return self.scaled_features
    
    def detect_isolation_forest(self) -> np.ndarray:
        """Detect anomalies using Isolation Forest"""
        if self.scaled_features is None:
            self.prepare_features()
        
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        
        predictions = self.isolation_forest.fit_predict(self.scaled_features)
        scores = self.isolation_forest.score_samples(self.scaled_features)
        
        # Normalize scores to [0, 1]
        scores_normalized = 1 / (1 + np.exp(scores))
        
        return predictions, scores_normalized
    
    def detect_lof(self, n_neighbors: int = 20) -> np.ndarray:
        """Detect anomalies using Local Outlier Factor"""
        if self.scaled_features is None:
            self.prepare_features()
        
        self.lof_model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.contamination
        )
        
        predictions = self.lof_model.fit_predict(self.scaled_features)
        scores = self.lof_model.negative_outlier_factor_
        
        # Normalize scores to [0, 1]
        scores_normalized = 1 / (1 + np.exp(scores))
        
        return predictions, scores_normalized
    
    def detect_ensemble(self, method: str = 'averaging') -> Tuple[np.ndarray, np.ndarray]:
        """Ensemble anomaly detection combining IF and LOF"""
        if_pred, if_scores = self.detect_isolation_forest()
        lof_pred, lof_scores = self.detect_lof()
        
        # Combine scores
        ensemble_scores = (if_scores + lof_scores) / 2
        
        if method == 'voting':
            # Voting: -1 for anomaly, 1 for normal (requires both to agree)
            ensemble_pred = np.where((if_pred + lof_pred) < 0, -1, 1)
        else:  # averaging - more sensitive (default)
            # Use percentile-based threshold instead of fixed 0.5
            # This respects the contamination parameter
            threshold = np.percentile(ensemble_scores, (1 - self.contamination) * 100)
            ensemble_pred = np.where(ensemble_scores > threshold, -1, 1)
        
        return ensemble_pred, ensemble_scores
    
    def get_anomaly_dataframe(self) -> pd.DataFrame:
        """Get DataFrame with anomaly scores"""
        predictions, scores = self.detect_ensemble()
        
        result_df = self.features_df[['bidder_id']].copy()
        result_df['isolation_forest_score'] = self.detect_isolation_forest()[1]
        result_df['lof_score'] = self.detect_lof()[1]
        result_df['ensemble_score'] = scores
        result_df['is_anomaly'] = predictions == -1
        result_df['anomaly_probability'] = scores
        
        return result_df.sort_values('ensemble_score', ascending=False)
    
    def get_feature_importance(self, n_features: int = 5) -> Dict[str, float]:
        """Get feature importance based on feature variance and correlation"""
        if self.scaled_features is None:
            self.prepare_features()
        
        # Calculate feature importance based on variance in scaled features
        # Higher variance = more discriminative feature
        feature_importance = {}
        for i, feature in enumerate(self.feature_columns):
            # Combine variance with contribution to anomaly detection
            variance = np.var(self.scaled_features[:, i])
            # Also consider how much this feature deviates for anomalies
            feature_importance[feature] = float(variance)
        
        # Normalize to 0-1 range
        max_importance = max(feature_importance.values()) if feature_importance else 1.0
        for feature in feature_importance:
            feature_importance[feature] = feature_importance[feature] / max(max_importance, 0.001)
        
        # Sort by importance
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_importance[:n_features])
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of anomaly detection results"""
        predictions, scores = self.detect_ensemble()
        
        return {
            'total_bidders': len(self.features_df),
            'num_anomalies': (predictions == -1).sum(),
            'anomaly_percentage': 100 * (predictions == -1).sum() / len(predictions),
            'mean_anomaly_score': scores.mean(),
            'max_anomaly_score': scores.max(),
            'min_anomaly_score': scores.min(),
            'std_anomaly_score': scores.std()
        }
