"""
Data Generation Module
Generates realistic synthetic auction data with configurable collusion patterns
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Set
from datetime import datetime, timedelta
import random
from scipy.stats import lognorm, gamma, expon


class SyntheticAuctionDataGenerator:
    """Generate realistic synthetic coal auction data with sophisticated collusion patterns"""
    
    def __init__(self, seed: int = None, num_auctions: int = 100, num_bidders: int = 50):
        """Initialize data generator"""
        # Use provided seed or generate random seed for true randomness
        self.seed = seed if seed is not None else np.random.randint(0, 2**31 - 1)
        self.num_auctions = num_auctions
        self.num_bidders = num_bidders
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        self.auctions_df = None
        self.bidders_df = None
        self.bids_df = None
        self.collusion_groups = None
        self.bidder_strategies = {}  # Track bidder strategies
        self.market_conditions = {}  # Track market variations
    
    def generate_bidders(self) -> pd.DataFrame:
        """Generate realistic bidder profiles with industry characteristics"""
        bidders = []
        
        # Bidder type distribution (realistic: more small, fewer large)
        num_large = max(1, self.num_bidders // 10)
        num_medium = max(2, self.num_bidders // 4)
        num_small = self.num_bidders - num_large - num_medium
        
        bidder_types = ['Large'] * num_large + ['Medium'] * num_medium + ['Small'] * num_small
        np.random.shuffle(bidder_types)
        
        for bidder_id in range(1, self.num_bidders + 1):
            bidder_type = bidder_types[bidder_id - 1]
            
            # Realistic capacity based on bidder type
            if bidder_type == 'Large':
                capacity = np.random.lognormal(mean=np.log(3000), sigma=0.4)
            elif bidder_type == 'Medium':
                capacity = np.random.lognormal(mean=np.log(1000), sigma=0.4)
            else:  # Small
                capacity = np.random.lognormal(mean=np.log(300), sigma=0.4)
            
            region = np.random.choice(['North', 'South', 'East', 'West', 'Central'], p=[0.25, 0.25, 0.2, 0.15, 0.15])
            
            # Create bidder strategy (honest vs aggressive vs collusive-prone)
            strategy = np.random.choice(['honest', 'aggressive', 'collusive-prone'], p=[0.6, 0.25, 0.15])
            
            bidder = {
                'bidder_id': bidder_id,
                'bidder_name': f'Company_{bidder_id:03d}',
                'bidder_type': bidder_type,
                'region': region,
                'capacity_tons': float(capacity),
                'efficiency_factor': np.random.uniform(0.7, 1.3),  # Bidding efficiency
                'aggressiveness': np.random.uniform(0.5, 1.5),  # How aggressively they bid
                'registration_date': datetime.now() - timedelta(days=np.random.randint(100, 1500)),
                'strategy': strategy,
                'experience_level': np.random.choice(['novice', 'experienced', 'expert'], p=[0.2, 0.5, 0.3]),
            }
            
            # Store strategy for later use
            self.bidder_strategies[bidder_id] = strategy
            
            bidders.append(bidder)
        
        self.bidders_df = pd.DataFrame(bidders)
        return self.bidders_df
    
    def generate_auctions(self) -> pd.DataFrame:
        """Generate realistic auction records with market conditions"""
        auctions = []
        base_date = datetime.now() - timedelta(days=365)
        
        # Market trend: prices may increase or decrease over time
        market_trend = np.random.uniform(-0.05, 0.05)  # -5% to +5% trend per auction
        
        for auction_id in range(1, self.num_auctions + 1):
            # Realistic coal quantity (lognormal distribution)
            coal_quantity = np.random.lognormal(mean=np.log(2000), sigma=0.8)
            
            # Base price with market trend
            base_price = 3500 * (1 + market_trend * (auction_id / self.num_auctions))
            reserve_price = base_price + np.random.normal(0, 200)
            opening_bid = reserve_price * np.random.uniform(1.0, 1.05)
            
            # Time of auction (more auctions during certain seasons)
            auction_date = base_date + timedelta(days=np.random.randint(0, 365))
            
            # Market condition for this auction
            market_competition = np.random.choice(['high', 'normal', 'low'], p=[0.3, 0.5, 0.2])
            
            auction = {
                'auction_id': auction_id,
                'auction_date': auction_date,
                'day_of_week': auction_date.weekday(),  # 0=Monday, 6=Sunday
                'month': auction_date.month,
                'coal_quantity_tons': float(coal_quantity),
                'coal_grade': np.random.choice(['A', 'B', 'C'], p=[0.3, 0.5, 0.2]),  # Quality grades
                'reserve_price': float(reserve_price),
                'opening_bid': float(opening_bid),
                'auction_status': np.random.choice(['Completed', 'Cancelled'], p=[0.95, 0.05]),
                'market_competition': market_competition,
                'base_price_trend': float(base_price),
                'buyer_region': np.random.choice(['North', 'South', 'East', 'West', 'Central']),
            }
            
            self.market_conditions[auction_id] = market_competition
            
            auctions.append(auction)
        
        self.auctions_df = pd.DataFrame(auctions)
        return self.auctions_df
    
    def create_collusion_groups(self, num_groups: int = 3) -> Dict[int, List[int]]:
        """Create realistic collusion groups with different patterns"""
        collusion_groups = {}
        
        # Collusive bidders are more likely to be collusive-prone or aggressive
        collusive_candidates = [
            bid_id for bid_id, strategy in self.bidder_strategies.items()
            if strategy in ['collusive-prone', 'aggressive']
        ]
        
        # Shuffle and divide into groups
        random.shuffle(collusive_candidates)
        
        for group_id in range(num_groups):
            # Vary group size (3-8 members)
            group_size = np.random.randint(3, 8)
            start_idx = group_id * (len(collusive_candidates) // num_groups)
            end_idx = start_idx + group_size
            
            if end_idx <= len(collusive_candidates):
                group_members = collusive_candidates[start_idx:end_idx]
                if len(group_members) >= 2:  # Only add if at least 2 members
                    collusion_groups[group_id] = group_members
        
        self.collusion_groups = collusion_groups
        return collusion_groups
    
    def generate_bids(self, collusion_percentage: float = 0.25) -> pd.DataFrame:
        """Generate realistic bids with sophisticated collusion patterns"""
        bids = []
        collusive_bidders = self._get_collusive_bidders()
        
        for auction_id in range(1, self.num_auctions + 1):
            auction_data = self.auctions_df[self.auctions_df['auction_id'] == auction_id].iloc[0]
            
            # Determine number of bidders for this auction based on market competition
            market_comp = self.market_conditions.get(auction_id, 'normal')
            if market_comp == 'high':
                num_bidders_auction = np.random.randint(8, 15)
            elif market_comp == 'low':
                num_bidders_auction = np.random.randint(2, 6)
            else:  # normal
                num_bidders_auction = np.random.randint(5, 12)
            
            # Decide if this auction has collusion
            is_collusive_auction = np.random.random() < collusion_percentage
            
            # Select bidders
            available_bidders = list(range(1, self.num_bidders + 1))
            selected_bidders = np.random.choice(available_bidders, min(num_bidders_auction, self.num_bidders), replace=False)
            
            # If collusive auction, increase probability of collusive bidders
            if is_collusive_auction and len(collusive_bidders) > 0:
                num_collusive = np.random.randint(2, min(4, len(collusive_bidders)))
                collusive_in_auction = list(np.random.choice(list(collusive_bidders), num_collusive, replace=False))
                selected_bidders = np.concatenate([collusive_in_auction[:len(selected_bidders)//2], 
                                                   selected_bidders[:len(selected_bidders)//2]])
            
            # Determine bidding prices and patterns
            opening_price = auction_data['opening_bid']
            
            # Create bid sequence
            current_bid_price = opening_price
            bid_times = self._generate_bid_times(len(selected_bidders))
            
            # Sort selected bidders to ensure price monotonicity
            sorted_bidders = sorted(selected_bidders)
            
            for position, bidder_id in enumerate(sorted_bidders):
                bidder_data = self.bidders_df[self.bidders_df['bidder_id'] == bidder_id].iloc[0]
                
                # Base bid price increment (monotonically increasing)
                if position == 0:
                    bid_price = opening_price
                else:
                    # Realistic increment pattern (usually 1-3% increments)
                    increment_pct = np.random.uniform(0.01, 0.03)
                    bid_price = current_bid_price * (1 + increment_pct)
                
                # Bidder-specific adjustments
                bid_price *= bidder_data['efficiency_factor']
                
                # COLLUSION PATTERNS
                if is_collusive_auction and bidder_id in collusive_bidders:
                    colluder_role = self._determine_colluder_role(position, len(sorted_bidders))
                    
                    if colluder_role == 'aggressive':
                        # Aggressive colluder - bids very high
                        bid_price *= np.random.uniform(1.08, 1.18)
                    elif colluder_role == 'supporter':
                        # Supporter - places bids to create false competition
                        bid_price *= np.random.uniform(1.02, 1.08)
                    elif colluder_role == 'silent':
                        # Silent colluder - bids but withdraws or bids passively
                        bid_price *= np.random.uniform(0.98, 1.02)
                
                # Market-based adjustment
                if market_comp == 'low':
                    bid_price *= np.random.uniform(0.95, 1.00)  # Lower prices in low competition
                elif market_comp == 'high':
                    bid_price *= np.random.uniform(1.00, 1.05)  # Higher prices in high competition
                
                # Ensure minimum price and maintain monotonicity
                bid_price = max(bid_price, current_bid_price)
                bid_price = max(bid_price, auction_data['reserve_price'] * 0.95)
                
                # Determine if bid is withdrawn (collusive bidders more likely to withdraw)
                withdrawal_prob = 0.05
                if is_collusive_auction and bidder_id in collusive_bidders:
                    withdrawal_prob = np.random.uniform(0.15, 0.35)  # Higher withdrawal for colluders
                
                bid_record = {
                    'auction_id': auction_id,
                    'bidder_id': bidder_id,
                    'bid_price': float(bid_price),
                    'bid_time': bid_times[position],
                    'bid_position': position,
                    'bid_increment_pct': (bid_price - opening_price) / opening_price if opening_price > 0 else 0,
                    'is_winning_bid': False,  # Will be set later
                    'bid_withdrawal': np.random.random() < withdrawal_prob,
                    'bidder_type': bidder_data['bidder_type'],
                    'bidder_region': bidder_data['region'],
                    'bidder_strategy': bidder_data['strategy'],
                    'is_collusive_bidder': bidder_id in collusive_bidders,
                    'is_collusive_auction': is_collusive_auction,
                    'market_competition': market_comp,
                }
                
                bids.append(bid_record)
                current_bid_price = bid_price
            
            # Mark winning bid (highest non-withdrawn bid)
            auction_bids = [b for b in bids if b['auction_id'] == auction_id]
            non_withdrawn = [b for b in auction_bids if not b['bid_withdrawal']]
            if non_withdrawn:
                winning_bid = max(non_withdrawn, key=lambda x: x['bid_price'])
                winning_bid['is_winning_bid'] = True
        
        self.bids_df = pd.DataFrame(bids)
        return self.bids_df
    
    def _generate_bid_times(self, num_bids: int) -> List[datetime]:
        """Generate realistic bid times throughout auction period"""
        auction_duration_hours = np.random.uniform(2, 24)  # Auction lasts 2-24 hours
        
        bid_times = []
        base_time = datetime.now() - timedelta(hours=auction_duration_hours)
        
        # Bids tend to cluster near the end (last hour)
        for i in range(num_bids):
            # 70% of bids in last 30% of time
            if np.random.random() < 0.7:
                time_offset = np.random.exponential(scale=auction_duration_hours * 0.1)
                time_offset = min(time_offset, auction_duration_hours * 0.3)
            else:
                time_offset = np.random.uniform(0, auction_duration_hours * 0.7)
            
            bid_time = base_time + timedelta(hours=time_offset)
            bid_times.append(bid_time)
        
        return sorted(bid_times)
    
    def _determine_colluder_role(self, position: int, total_bidders: int) -> str:
        """Determine the role a colluder plays in this auction"""
        # Roles: 'aggressive' (places winning bid), 'supporter' (creates fake competition), 'silent' (withdrawn)
        if position == 0:
            return np.random.choice(['aggressive', 'silent'], p=[0.7, 0.3])
        elif position < total_bidders // 2:
            return np.random.choice(['supporter', 'aggressive'], p=[0.6, 0.4])
        else:
            return np.random.choice(['supporter', 'silent'], p=[0.5, 0.5])
    
    def _get_collusive_bidders(self) -> Set[int]:
        """Get set of all bidders in collusion groups"""
        if not self.collusion_groups:
            return set()
        return set(bidder for group in self.collusion_groups.values() for bidder in group)
    
    def export_data_for_analysis(self) -> Dict[str, pd.DataFrame]:
        """Export data in formats useful for analysis"""
        if self.bids_df is None:
            return {}
        
        # Aggregate bidder statistics
        bidder_stats = self.bids_df.groupby('bidder_id').agg({
            'auction_id': 'nunique',
            'bid_price': ['mean', 'median', 'std', 'min', 'max'],
            'is_winning_bid': 'sum',
            'bid_withdrawal': 'sum'
        }).reset_index()
        
        bidder_stats.columns = ['bidder_id', 'auctions_participated', 'avg_bid_price', 'median_bid_price', 
                               'bid_price_std', 'min_bid', 'max_bid', 'wins', 'withdrawals']
        bidder_stats['win_rate'] = bidder_stats['wins'] / bidder_stats['auctions_participated']
        bidder_stats['withdrawal_rate'] = bidder_stats['withdrawals'] / (len(self.bids_df[self.bids_df['bidder_id'].isin(bidder_stats['bidder_id'])]) / len(bidder_stats))
        
        # Auction aggregates
        auction_stats = self.bids_df.groupby('auction_id').agg({
            'bidder_id': 'nunique',
            'bid_price': ['min', 'max', 'mean'],
            'bid_withdrawal': 'sum',
            'is_collusive_bidder': 'sum'
        }).reset_index()
        
        auction_stats.columns = ['auction_id', 'num_bidders', 'min_price', 'max_price', 'avg_price', 'withdrawals', 'collusive_bidders_count']
        
        return {
            'auctions': self.auctions_df,
            'bidders': self.bidders_df,
            'bids': self.bids_df,
            'bidder_stats': bidder_stats,
            'auction_stats': auction_stats
        }
    
    def generate_full_dataset(self, num_auctions: int = None, num_bidders: int = None, 
                            num_collusion_groups: int = 3, collusion_percentage: float = 0.25) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate complete dataset"""
        if num_auctions:
            self.num_auctions = num_auctions
        if num_bidders:
            self.num_bidders = num_bidders
        
        self.generate_bidders()
        self.generate_auctions()
        self.create_collusion_groups(num_collusion_groups)
        self.generate_bids(collusion_percentage)
        
        return self.auctions_df, self.bidders_df, self.bids_df
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics"""
        if self.bids_df is None:
            return {}
        
        # Calculate various metrics
        non_withdrawn_bids = self.bids_df[~self.bids_df['bid_withdrawal']]
        winning_bids = self.bids_df[self.bids_df['is_winning_bid']]
        collusive_bids = self.bids_df[self.bids_df['is_collusive_bidder']]
        collusive_auction_bids = self.bids_df[self.bids_df['is_collusive_auction']]
        
        return {
            'total_auctions': len(self.auctions_df),
            'total_bidders': len(self.bidders_df),
            'total_bids': len(self.bids_df),
            'total_collusion_groups': len(self.collusion_groups) if self.collusion_groups else 0,
            'bidders_in_collusion': len(self._get_collusive_bidders()),
            
            'avg_bids_per_auction': self.bids_df.groupby('auction_id').size().mean(),
            'median_bids_per_auction': self.bids_df.groupby('auction_id').size().median(),
            'avg_bid_price': self.bids_df['bid_price'].mean(),
            'median_bid_price': self.bids_df['bid_price'].median(),
            'bid_price_std': self.bids_df['bid_price'].std(),
            'bid_price_range': (self.bids_df['bid_price'].max() - self.bids_df['bid_price'].min()),
            
            'avg_bid_increment_pct': self.bids_df['bid_increment_pct'].mean(),
            'avg_bid_withdrawal_rate': self.bids_df['bid_withdrawal'].mean(),
            'collusive_bids_count': len(collusive_bids),
            'collusive_auction_count': self.bids_df['is_collusive_auction'].sum() // max(self.bids_df.groupby('auction_id').size().mean(), 1),
            
            'winning_bids_avg_price': winning_bids['bid_price'].mean() if len(winning_bids) > 0 else 0,
            'average_auction_quantity': self.auctions_df['coal_quantity_tons'].mean(),
            'total_coal_volume': self.auctions_df['coal_quantity_tons'].sum(),
            
            'cancelled_auctions': (self.auctions_df['auction_status'] == 'Cancelled').sum(),
            'completed_auctions': (self.auctions_df['auction_status'] == 'Completed').sum(),
        }
    
    def validate_data(self) -> Dict[str, Any]:
        """Validate data consistency and quality"""
        issues = []
        warnings = []
        
        # Check for duplicate bids at same price by same bidder
        for auction_id in self.bids_df['auction_id'].unique():
            auction_bids = self.bids_df[self.bids_df['auction_id'] == auction_id]
            if len(auction_bids) < 2:
                warnings.append(f"Auction {auction_id} has less than 2 bids")
            
            # Check bid progression
            non_withdrawn = auction_bids[~auction_bids['bid_withdrawal']]
            if len(non_withdrawn) > 1:
                prices = non_withdrawn['bid_price'].values
                if not all(prices[i] <= prices[i+1] for i in range(len(prices)-1)):
                    issues.append(f"Auction {auction_id}: Non-monotonic bid prices")
        
        # Check for bidders with no participation
        bidders_in_bids = set(self.bids_df['bidder_id'].unique())
        all_bidders = set(self.bidders_df['bidder_id'].unique())
        non_participating = all_bidders - bidders_in_bids
        if len(non_participating) > 0:
            warnings.append(f"Found {len(non_participating)} bidders with no bids")
        
        # Check collusion groups validity
        for group_id, members in self.collusion_groups.items():
            if len(members) < 2:
                issues.append(f"Collusion group {group_id} has less than 2 members")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'stats': {
                'non_participating_bidders': len(non_participating),
                'total_issues': len(issues),
                'total_warnings': len(warnings)
            }
        }
    
    def get_collusion_report(self) -> Dict[str, Any]:
        """Generate detailed collusion analysis report"""
        collusive_bidders = self._get_collusive_bidders()
        
        report = {
            'total_collusion_groups': len(self.collusion_groups),
            'collusion_groups': self.collusion_groups,
            'bidders_in_collusion': list(collusive_bidders),
            'collusion_group_sizes': {gid: len(members) for gid, members in self.collusion_groups.items()},
        }
        
        # Analyze collusive auction patterns
        collusive_auctions = self.bids_df[self.bids_df['is_collusive_auction']]
        if len(collusive_auctions) > 0:
            report['collusive_auctions_count'] = collusive_auctions['auction_id'].nunique()
            report['avg_colluders_per_auction'] = collusive_auctions.groupby('auction_id')['bidder_id'].apply(
                lambda x: len([b for b in x if b in collusive_bidders])
            ).mean()
            
            # Collusive bidder success rate
            collusive_winners = collusive_auctions[
                (collusive_auctions['is_winning_bid']) & 
                (collusive_auctions['bidder_id'].isin(collusive_bidders))
            ]
            report['collusive_bidders_win_rate'] = len(collusive_winners) / len(collusive_auctions.groupby('auction_id')) if len(collusive_auctions.groupby('auction_id')) > 0 else 0
        
        return report
