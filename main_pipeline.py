"""
Main Pipeline Module
Orchestrates the complete collusion detection workflow
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import pandas as pd

from data_generator import SyntheticAuctionDataGenerator
from feature_engineering import FeatureEngineer
from anomaly_detection import AnomalyDetector
from network_analysis import BidderNetworkAnalysis
from risk_scoring import CollusionRiskScorer
from config import DetectionConfig, load_config


class CollusionDetectionPipeline:
    """Main pipeline orchestrating collusion detection"""
    
    def __init__(self, config: Optional[DetectionConfig] = None, output_dir: str = './results'):
        """Initialize pipeline"""
        self.config = config or DetectionConfig()
        self.output_dir = output_dir
        self.setup_logging()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Components
        self.data_generator = None
        self.feature_engineer = None
        self.anomaly_detector = None
        self.network_analyzer = None
        self.risk_scorer = None
        
        # Data
        self.auctions_df = None
        self.bidders_df = None
        self.bids_df = None
        self.features_df = None
        self.anomaly_scores_df = None
        self.risk_scores_df = None
    
    def setup_logging(self) -> None:
        """Setup logging"""
        log_file = os.path.join(self.output_dir, f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def step_1_generate_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Step 1: Generate synthetic auction data"""
        self.logger.info("=" * 80)
        self.logger.info("STEP 1: Generating Synthetic Auction Data")
        self.logger.info("=" * 80)
        
        self.data_generator = SyntheticAuctionDataGenerator(
            seed=self.config.seed,
            num_auctions=self.config.num_auctions,
            num_bidders=self.config.num_bidders
        )
        
        self.auctions_df, self.bidders_df, self.bids_df = self.data_generator.generate_full_dataset(
            num_auctions=self.config.num_auctions,
            num_bidders=self.config.num_bidders,
            num_collusion_groups=3,
            collusion_percentage=self.config.collusion_percentage
        )
        
        summary = self.data_generator.get_summary_statistics()
        self.logger.info(f"Generated data: {summary}")
        self.logger.info(f"Auctions: {len(self.auctions_df)}, Bidders: {len(self.bidders_df)}, Bids: {len(self.bids_df)}")
        
        return self.auctions_df, self.bidders_df, self.bids_df
    
    def step_2_extract_features(self) -> pd.DataFrame:
        """Step 2: Extract features"""
        self.logger.info("=" * 80)
        self.logger.info("STEP 2: Extracting Features")
        self.logger.info("=" * 80)
        
        self.feature_engineer = FeatureEngineer(self.auctions_df, self.bidders_df, self.bids_df)
        self.features_df = self.feature_engineer.combine_all_features()
        
        self.logger.info(f"Extracted {len(self.features_df.columns) - 1} features for {len(self.features_df)} bidders")
        
        feature_stats = self.feature_engineer.get_feature_statistics()
        self.logger.info(f"Feature statistics calculated")
        
        return self.features_df
    
    def step_3_detect_anomalies(self) -> pd.DataFrame:
        """Step 3: Detect anomalies"""
        self.logger.info("=" * 80)
        self.logger.info("STEP 3: Detecting Anomalies")
        self.logger.info("=" * 80)
        
        self.anomaly_detector = AnomalyDetector(
            self.features_df,
            contamination=self.config.isolation_forest_contamination
        )
        
        self.anomaly_scores_df = self.anomaly_detector.get_anomaly_dataframe()
        
        summary = self.anomaly_detector.get_anomaly_summary()
        self.logger.info(f"Anomaly detection summary: {summary}")
        
        return self.anomaly_scores_df
    
    def step_4_analyze_network(self) -> Dict[int, Dict[str, float]]:
        """Step 4: Analyze bidder network"""
        self.logger.info("=" * 80)
        self.logger.info("STEP 4: Analyzing Bidder Network")
        self.logger.info("=" * 80)
        
        self.network_analyzer = BidderNetworkAnalysis(
            self.bids_df,
            min_coprediction_threshold=self.config.min_coprediction_threshold
        )
        
        self.network_analyzer.build_cobidding_network()
        self.network_analyzer.detect_communities()
        
        summary = self.network_analyzer.get_network_analysis_summary()
        self.logger.info(f"Network analysis summary: {summary}")
        
        # Calculate network scores for all bidders
        network_scores = {}
        for bidder_id in self.features_df['bidder_id']:
            network_scores[bidder_id] = self.network_analyzer.get_bidder_network_score(bidder_id)
        
        return network_scores
    
    def step_5_score_risks(self, network_scores: Dict[int, Dict[str, float]]) -> pd.DataFrame:
        """Step 5: Calculate risk scores"""
        self.logger.info("=" * 80)
        self.logger.info("STEP 5: Scoring Collusion Risk")
        self.logger.info("=" * 80)
        
        config_dict = {
            'anomaly_score_weight': self.config.anomaly_score_weight,
            'network_score_weight': self.config.network_score_weight,
            'behavior_score_weight': self.config.behavior_score_weight,
            'risk_threshold_low': self.config.risk_threshold_low,
            'risk_threshold_medium': self.config.risk_threshold_medium,
            'risk_threshold_high': self.config.risk_threshold_high
        }
        
        self.risk_scorer = CollusionRiskScorer(
            self.features_df,
            self.anomaly_scores_df,
            network_scores,
            config_dict
        )
        
        self.risk_scores_df = self.risk_scorer.calculate_all_risk_scores()
        
        summary = self.risk_scorer.get_scoring_summary()
        self.logger.info(f"Risk scoring summary: {summary}")
        
        return self.risk_scores_df
    
    def step_6_save_results(self) -> None:
        """Step 6: Save results"""
        self.logger.info("=" * 80)
        self.logger.info("STEP 6: Saving Results")
        self.logger.info("=" * 80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.config.save_intermediate_results:
            self.auctions_df.to_csv(os.path.join(self.output_dir, f'auctions_{timestamp}.csv'), index=False)
            self.bidders_df.to_csv(os.path.join(self.output_dir, f'bidders_{timestamp}.csv'), index=False)
            self.bids_df.to_csv(os.path.join(self.output_dir, f'bids_{timestamp}.csv'), index=False)
            self.features_df.to_csv(os.path.join(self.output_dir, f'features_{timestamp}.csv'), index=False)
            self.anomaly_scores_df.to_csv(os.path.join(self.output_dir, f'anomaly_scores_{timestamp}.csv'), index=False)
        
        self.risk_scores_df.to_csv(os.path.join(self.output_dir, f'risk_scores_{timestamp}.csv'), index=False)
        
        self.logger.info(f"Results saved to {self.output_dir}")
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Run complete pipeline"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STARTING COLLUSION DETECTION PIPELINE")
        self.logger.info("=" * 80 + "\n")
        
        # Execute steps
        self.step_1_generate_data()
        self.step_2_extract_features()
        self.step_3_detect_anomalies()
        network_scores = self.step_4_analyze_network()
        self.step_5_score_risks(network_scores)
        self.step_6_save_results()
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info("=" * 80 + "\n")
        
        return {
            'risk_scores': self.risk_scores_df,
            'auctions': self.auctions_df,
            'bidders': self.bidders_df,
            'bids': self.bids_df,
            'features': self.features_df,
            'anomaly_scores': self.anomaly_scores_df
        }


def main():
    """Main entry point"""
    # Load configuration
    config = load_config(preset_name='balanced')
    
    # Create and run pipeline
    pipeline = CollusionDetectionPipeline(config=config, output_dir='./results')
    results = pipeline.run_pipeline()
    
    # Print high-risk bidders
    high_risk = results['risk_scores'][results['risk_scores']['risk_level'] == 'HIGH']
    print(f"\n\nHIGH-RISK BIDDERS ({len(high_risk)}):")
    print(high_risk[['bidder_id', 'combined_risk_score', 'risk_level']])


if __name__ == '__main__':
    main()
