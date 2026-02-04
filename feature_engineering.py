"""
Feature Engineering Module
Extracts comprehensive features from auction data for collusion detection
Implements 40+ features across participation, pricing, temporal, and behavioral patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Extract and engineer features for collusion detection"""
    
    def __init__(self, auctions_df: pd.DataFrame, bidders_df: pd.DataFrame, bids_df: pd.DataFrame):
        """Initialize feature engineer"""
        self.auctions_df = auctions_df.copy()
        self.bidders_df = bidders_df.copy()
        self.bids_df = bids_df.copy()
        self.features_df = None
    
    def extract_bidder_features(self) -> pd.DataFrame:
        """Extract comprehensive bidder-level features (participation, pricing, winning)"""
        bidder_features = []
        
        for bidder_id in self.bidders_df['bidder_id']:
            bidder_info = self.bidders_df[self.bidders_df['bidder_id'] == bidder_id].iloc[0]
            bidder_bids = self.bids_df[self.bids_df['bidder_id'] == bidder_id]
            
            if len(bidder_bids) == 0:
                continue
            
            features = {'bidder_id': bidder_id}
            
            # ===== PARTICIPATION FEATURES =====
            features['num_auctions'] = bidder_bids['auction_id'].nunique()
            features['num_bids'] = len(bidder_bids)
            features['total_bids_placed'] = len(bidder_bids)
            features['avg_bids_per_auction'] = features['num_bids'] / max(features['num_auctions'], 1)
            features['participation_rate'] = features['num_auctions'] / len(self.auctions_df)
            features['bidder_type_encoded'] = {'Large': 3, 'Medium': 2, 'Small': 1}.get(bidder_info['bidder_type'], 1)
            
            # ===== PRICE FEATURES =====
            features['avg_bid_price'] = bidder_bids['bid_price'].mean()
            features['median_bid_price'] = bidder_bids['bid_price'].median()
            features['bid_price_std'] = bidder_bids['bid_price'].std() or 0
            features['bid_price_cv'] = features['bid_price_std'] / max(features['avg_bid_price'], 1)  # Coefficient of variation
            features['bid_price_skewness'] = bidder_bids['bid_price'].skew() or 0  # Price distribution skewness
            features['bid_price_kurtosis'] = bidder_bids['bid_price'].kurtosis() or 0  # Price distribution kurtosis
            features['max_bid_price'] = bidder_bids['bid_price'].max()
            features['min_bid_price'] = bidder_bids['bid_price'].min()
            features['bid_price_range'] = features['max_bid_price'] - features['min_bid_price']
            features['bid_price_quartile_range'] = bidder_bids['bid_price'].quantile(0.75) - bidder_bids['bid_price'].quantile(0.25)
            
            # ===== WINNING FEATURES =====
            features['num_wins'] = bidder_bids['is_winning_bid'].sum()
            features['win_rate'] = bidder_bids['is_winning_bid'].sum() / max(len(bidder_bids), 1)
            features['win_price_avg'] = bidder_bids[bidder_bids['is_winning_bid']]['bid_price'].mean() or 0
            features['loss_price_avg'] = bidder_bids[~bidder_bids['is_winning_bid']]['bid_price'].mean() or 0
            features['win_loss_price_ratio'] = features['win_price_avg'] / max(features['loss_price_avg'], 1)
            features['consecutive_losses'] = self._calculate_consecutive_losses(bidder_bids)
            features['consecutive_wins'] = self._calculate_consecutive_wins(bidder_bids)
            
            # ===== WITHDRAWAL FEATURES =====
            features['withdrawal_rate'] = bidder_bids['bid_withdrawal'].sum() / max(len(bidder_bids), 1)
            features['num_withdrawals'] = bidder_bids['bid_withdrawal'].sum()
            features['withdrawals_after_leading'] = self._count_withdrawals_after_leading(bidder_bids)
            features['withdrawal_before_loss_rate'] = self._withdrawal_before_loss_rate(bidder_bids)
            
            # ===== BID POSITION FEATURES =====
            features['avg_bid_position'] = bidder_bids['bid_position'].mean()
            features['median_bid_position'] = bidder_bids['bid_position'].median()
            features['first_bid_ratio'] = (bidder_bids['bid_position'] == 0).sum() / max(len(bidder_bids), 1)
            features['last_bid_ratio'] = self._last_bid_ratio(bidder_bids)
            features['early_bid_ratio'] = ((bidder_bids['bid_position'] < bidder_bids['bid_position'].median()) & (bidder_bids['bid_position'] > 0)).sum() / max(len(bidder_bids), 1)
            
            # ===== INCREMENT FEATURES =====
            features['avg_bid_increment_pct'] = bidder_bids['bid_increment_pct'].mean()
            features['std_bid_increment_pct'] = bidder_bids['bid_increment_pct'].std() or 0
            features['max_bid_increment_pct'] = bidder_bids['bid_increment_pct'].max()
            features['min_bid_increment_pct'] = bidder_bids['bid_increment_pct'].min()
            features['bid_increment_consistency'] = 1 / (1 + features['std_bid_increment_pct']) if features['std_bid_increment_pct'] > 0 else 1
            
            # ===== BIDDER CHARACTERISTICS =====
            features['capacity'] = bidder_info['capacity_tons']
            features['efficiency_factor'] = bidder_info['efficiency_factor']
            features['aggressiveness'] = bidder_info['aggressiveness']
            features['strategy_honest'] = 1 if bidder_info['strategy'] == 'honest' else 0
            features['strategy_aggressive'] = 1 if bidder_info['strategy'] == 'aggressive' else 0
            features['strategy_collusive'] = 1 if bidder_info['strategy'] == 'collusive-prone' else 0
            
            bidder_features.append(features)
        
        return pd.DataFrame(bidder_features)
    
    def extract_temporal_features(self) -> pd.DataFrame:
        """Extract temporal and pattern-based features"""
        temporal_features = []
        
        for bidder_id in self.bidders_df['bidder_id']:
            bidder_bids = self.bids_df[self.bids_df['bidder_id'] == bidder_id].copy()
            
            if len(bidder_bids) < 2:
                continue
            
            bidder_bids = bidder_bids.sort_values('bid_time').reset_index(drop=True)
            
            features = {'bidder_id': bidder_id}
            
            # ===== TEMPORAL CONSISTENCY =====
            time_diffs = bidder_bids['bid_time'].diff().dt.total_seconds()
            features['avg_bidding_interval_sec'] = time_diffs.mean() or 0
            features['std_bidding_interval_sec'] = time_diffs.std() or 0
            features['min_bidding_interval_sec'] = time_diffs.min() or 0
            features['max_bidding_interval_sec'] = time_diffs.max() or 0
            features['bidding_regularity'] = 1 / (1 + features['std_bidding_interval_sec']) if features['std_bidding_interval_sec'] > 0 else 1
            
            # ===== HOUR PATTERNS =====
            hour_of_day = bidder_bids['bid_time'].dt.hour
            hour_counts = hour_of_day.value_counts()
            features['preferred_bidding_hour'] = hour_counts.idxmax() if len(hour_counts) > 0 else -1
            features['bidding_hour_concentration'] = hour_counts.max() / len(bidder_bids)  # How concentrated in preferred hour
            features['bidding_hours_used'] = len(hour_counts)  # How many different hours they bid in
            features['bidding_hour_entropy'] = stats.entropy(hour_counts) if len(hour_counts) > 1 else 0
            
            # ===== DAY OF WEEK PATTERNS =====
            day_of_week = bidder_bids['bid_time'].dt.dayofweek
            dow_counts = day_of_week.value_counts()
            features['preferred_day_of_week'] = dow_counts.idxmax() if len(dow_counts) > 0 else -1
            features['bidding_day_concentration'] = dow_counts.max() / len(bidder_bids)
            features['bidding_days_used'] = len(dow_counts)
            
            # ===== LATE BID TENDENCY (Last-minute bidding) =====
            features['late_bid_ratio'] = self._calculate_late_bid_ratio(bidder_bids)
            features['very_late_bid_ratio'] = self._calculate_very_late_bid_ratio(bidder_bids)
            
            # ===== BIDDING FREQUENCY PATTERNS =====
            features['avg_bids_per_day'] = self._calculate_bids_per_day(bidder_bids)
            features['bids_concentration_day'] = self._calculate_bids_concentration_by_day(bidder_bids)
            
            temporal_features.append(features)
        
        return pd.DataFrame(temporal_features)
    
    def extract_price_features(self) -> pd.DataFrame:
        """Extract auction-level price statistics for bidders"""
        price_features = []
        
        for bidder_id in self.bidders_df['bidder_id']:
            bidder_auctions = self.bids_df[self.bids_df['bidder_id'] == bidder_id]
            
            if len(bidder_auctions) == 0:
                continue
            
            features = {'bidder_id': bidder_id}
            
            # ===== PRICE RELATIVE TO RESERVE =====
            auction_ids = bidder_auctions['auction_id'].unique()
            price_ratios = []
            for auction_id in auction_ids:
                auction_bids = bidder_auctions[bidder_auctions['auction_id'] == auction_id]
                auction_data = self.auctions_df[self.auctions_df['auction_id'] == auction_id].iloc[0]
                bid_prices = auction_bids['bid_price'].values
                reserve = auction_data['reserve_price']
                price_ratios.extend(bid_prices / reserve)
            
            features['avg_price_to_reserve_ratio'] = np.mean(price_ratios) if price_ratios else 1.0
            features['std_price_to_reserve_ratio'] = np.std(price_ratios) if len(price_ratios) > 1 else 0
            
            # ===== PRICE RELATIVE TO OPENING BID =====
            opening_ratios = []
            for auction_id in auction_ids:
                auction_bids = bidder_auctions[bidder_auctions['auction_id'] == auction_id]
                auction_data = self.auctions_df[self.auctions_df['auction_id'] == auction_id].iloc[0]
                bid_prices = auction_bids['bid_price'].values
                opening = auction_data['opening_bid']
                opening_ratios.extend(bid_prices / opening)
            
            features['avg_price_to_opening_ratio'] = np.mean(opening_ratios) if opening_ratios else 1.0
            
            # ===== WINNING BID CHARACTERISTICS =====
            winning_bids = bidder_auctions[bidder_auctions['is_winning_bid']]
            if len(winning_bids) > 0:
                features['winning_bids_count'] = len(winning_bids)
                features['avg_winning_bid_position'] = winning_bids['bid_position'].mean()
                features['winning_as_first_bid_ratio'] = (winning_bids['bid_position'] == 0).sum() / len(winning_bids)
            else:
                features['winning_bids_count'] = 0
                features['avg_winning_bid_position'] = -1
                features['winning_as_first_bid_ratio'] = 0
            
            price_features.append(features)
        
        return pd.DataFrame(price_features)
    
    def extract_network_features(self) -> pd.DataFrame:
        """Extract network-based co-bidding features"""
        network_features = []
        
        # Build auction-bidder network
        auction_bidders_map = {}
        for auction_id in self.bids_df['auction_id'].unique():
            auction_bidders_map[auction_id] = set(
                self.bids_df[self.bids_df['auction_id'] == auction_id]['bidder_id'].unique()
            )
        
        for bidder_id in self.bidders_df['bidder_id']:
            features = {'bidder_id': bidder_id}
            
            # ===== CO-BIDDING STATISTICS =====
            bidder_auctions = self.bids_df[self.bids_df['bidder_id'] == bidder_id]['auction_id'].unique()
            co_bidders_set = set()
            co_bidding_counts = {}
            
            for auction_id in bidder_auctions:
                other_bidders = auction_bidders_map[auction_id] - {bidder_id}
                co_bidders_set.update(other_bidders)
                for other_id in other_bidders:
                    co_bidding_counts[other_id] = co_bidding_counts.get(other_id, 0) + 1
            
            features['num_unique_co_bidders'] = len(co_bidders_set)
            features['co_bidder_concentration'] = len(co_bidders_set) / max(len(self.bidders_df) - 1, 1)
            
            # ===== CO-BIDDING FREQUENCY =====
            if co_bidding_counts:
                features['avg_co_bids_frequency'] = np.mean(list(co_bidding_counts.values()))
                features['max_co_bids_frequency'] = max(co_bidding_counts.values())
                features['co_bids_frequency_std'] = np.std(list(co_bidding_counts.values()))
                
                # Bidders that frequently co-bid with this bidder (potential collaborators)
                frequent_co_bidders = [k for k, v in co_bidding_counts.items() if v >= 3]
                features['frequent_co_bidders_count'] = len(frequent_co_bidders)
                features['frequent_co_bidding_ratio'] = len(frequent_co_bidders) / max(len(co_bidders_set), 1)
            else:
                features['avg_co_bids_frequency'] = 0
                features['max_co_bids_frequency'] = 0
                features['co_bids_frequency_std'] = 0
                features['frequent_co_bidders_count'] = 0
                features['frequent_co_bidding_ratio'] = 0
            
            # ===== NETWORK CLUSTERING =====
            features['network_density_local'] = self._calculate_local_network_density(bidder_id, co_bidders_set, auction_bidders_map)
            
            network_features.append(features)
        
        return pd.DataFrame(network_features)
    
    def combine_all_features(self) -> pd.DataFrame:
        """Combine all features into single comprehensive DataFrame"""
        bidder_features = self.extract_bidder_features()
        temporal_features = self.extract_temporal_features()
        price_features = self.extract_price_features()
        network_features = self.extract_network_features()
        
        # Merge all features
        self.features_df = bidder_features.copy()
        
        if len(temporal_features) > 0:
            self.features_df = self.features_df.merge(temporal_features, on='bidder_id', how='left')
        
        if len(price_features) > 0:
            self.features_df = self.features_df.merge(price_features, on='bidder_id', how='left')
        
        if len(network_features) > 0:
            self.features_df = self.features_df.merge(network_features, on='bidder_id', how='left')
        
        # Fill NaN values
        self.features_df = self.features_df.fillna(0)
        
        return self.features_df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        if self.features_df is None:
            self.combine_all_features()
        
        return [col for col in self.features_df.columns if col != 'bidder_id']
    
    def get_feature_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each feature"""
        if self.features_df is None:
            self.combine_all_features()
        
        stats_dict = {}
        for col in self.features_df.columns:
            if col != 'bidder_id':
                try:
                    # Convert to numeric if needed
                    col_data = pd.to_numeric(self.features_df[col], errors='coerce')
                    # Skip if all values are NaN
                    if col_data.isna().all():
                        continue
                    
                    stats_dict[col] = {
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'median': float(col_data.median()),
                        'q25': float(col_data.quantile(0.25)),
                        'q75': float(col_data.quantile(0.75))
                    }
                except (ValueError, TypeError):
                    # Skip columns that can't be converted to numeric
                    continue
        
        return stats_dict
    
    # ===== HELPER METHODS =====
    
    def _calculate_consecutive_losses(self, bidder_bids: pd.DataFrame) -> int:
        """Calculate longest consecutive losses"""
        wins = bidder_bids['is_winning_bid'].astype(int).values
        max_loss = 0
        current_loss = 0
        for win in wins:
            if win == 0:
                current_loss += 1
                max_loss = max(max_loss, current_loss)
            else:
                current_loss = 0
        return max_loss
    
    def _calculate_consecutive_wins(self, bidder_bids: pd.DataFrame) -> int:
        """Calculate longest consecutive wins"""
        wins = bidder_bids['is_winning_bid'].astype(int).values
        max_win = 0
        current_win = 0
        for win in wins:
            if win == 1:
                current_win += 1
                max_win = max(max_win, current_win)
            else:
                current_win = 0
        return max_win
    
    def _count_withdrawals_after_leading(self, bidder_bids: pd.DataFrame) -> int:
        """Count withdrawals that occurred after bidder had winning bid"""
        count = 0
        had_winning = False
        for idx, row in bidder_bids.iterrows():
            if row['is_winning_bid']:
                had_winning = True
            elif had_winning and row['bid_withdrawal']:
                count += 1
        return count
    
    def _withdrawal_before_loss_rate(self, bidder_bids: pd.DataFrame) -> float:
        """Rate of withdrawing before potentially losing"""
        withdrawal_count = 0
        potentially_losing = 0
        
        for idx, row in bidder_bids.iterrows():
            if not row['is_winning_bid']:
                potentially_losing += 1
                if row['bid_withdrawal']:
                    withdrawal_count += 1
        
        return withdrawal_count / max(potentially_losing, 1)
    
    def _last_bid_ratio(self, bidder_bids: pd.DataFrame) -> float:
        """Ratio of bids that were last position in auction"""
        last_positions = []
        for auction_id in bidder_bids['auction_id'].unique():
            auction_bids = bidder_bids[bidder_bids['auction_id'] == auction_id]
            max_pos = auction_bids['bid_position'].max()
            last_positions.append(max_pos)
        
        if not last_positions:
            return 0
        
        last_count = sum(bidder_bids['bid_position'] == pos for pos in last_positions)
        return last_count / len(bidder_bids)
    
    def _calculate_late_bid_ratio(self, bidder_bids: pd.DataFrame) -> float:
        """Ratio of bids in last 25% of auction period"""
        late_count = 0
        for auction_id in bidder_bids['auction_id'].unique():
            auction_bids = bidder_bids[bidder_bids['auction_id'] == auction_id]
            if len(auction_bids) > 0:
                time_range = auction_bids['bid_time'].max() - auction_bids['bid_time'].min()
                cutoff = auction_bids['bid_time'].min() + (time_range * 0.75)
                late_count += (auction_bids['bid_time'] >= cutoff).sum()
        
        return late_count / max(len(bidder_bids), 1)
    
    def _calculate_very_late_bid_ratio(self, bidder_bids: pd.DataFrame) -> float:
        """Ratio of bids in last 10% of auction period"""
        very_late_count = 0
        for auction_id in bidder_bids['auction_id'].unique():
            auction_bids = bidder_bids[bidder_bids['auction_id'] == auction_id]
            if len(auction_bids) > 0:
                time_range = auction_bids['bid_time'].max() - auction_bids['bid_time'].min()
                cutoff = auction_bids['bid_time'].min() + (time_range * 0.90)
                very_late_count += (auction_bids['bid_time'] >= cutoff).sum()
        
        return very_late_count / max(len(bidder_bids), 1)
    
    def _calculate_bids_per_day(self, bidder_bids: pd.DataFrame) -> float:
        """Average bids per day"""
        days = bidder_bids['bid_time'].dt.date.nunique()
        return len(bidder_bids) / max(days, 1)
    
    def _calculate_bids_concentration_by_day(self, bidder_bids: pd.DataFrame) -> float:
        """How concentrated bids are by day (0-1, 1 = all in one day)"""
        day_counts = bidder_bids['bid_time'].dt.date.value_counts()
        if len(day_counts) == 0:
            return 0
        return day_counts.max() / len(bidder_bids)
    
    def _calculate_local_network_density(self, bidder_id: int, co_bidders: set, auction_bidders_map: dict) -> float:
        """Calculate local network density (how interconnected are co-bidders)"""
        if len(co_bidders) < 2:
            return 0
        
        # Count edges between co-bidders
        edge_count = 0
        bidder_auctions = self.bids_df[self.bids_df['bidder_id'] == bidder_id]['auction_id'].unique()
        
        for auction_id in bidder_auctions:
            auction_co_bidders = auction_bidders_map[auction_id] & co_bidders
            # Each pair that co-bids is an edge
            edge_count += len(auction_co_bidders) * (len(auction_co_bidders) - 1) / 2
        
        # Possible edges in complete graph
        max_edges = len(co_bidders) * (len(co_bidders) - 1) / 2
        
        return edge_count / max(max_edges, 1)
