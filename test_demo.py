"""
Testing and Demonstration Module
Provides examples and tests for the collusion detection system
"""

import sys
from data_generator import SyntheticAuctionDataGenerator
from feature_engineering import FeatureEngineer
from anomaly_detection import AnomalyDetector
from network_analysis import BidderNetworkAnalysis
from risk_scoring import CollusionRiskScorer
from visualizations import CollusionVisualizer
from main_pipeline import CollusionDetectionPipeline
from config import load_config


def test_data_generation():
    """Test data generation"""
    print("\n" + "="*80)
    print("TEST 1: Data Generation")
    print("="*80)
    
    generator = SyntheticAuctionDataGenerator(num_auctions=50, num_bidders=30)
    auctions, bidders, bids = generator.generate_full_dataset()
    
    print(f"✓ Generated {len(auctions)} auctions")
    print(f"✓ Generated {len(bidders)} bidders")
    print(f"✓ Generated {len(bids)} bids")
    
    summary = generator.get_summary_statistics()
    print(f"\nSummary: {summary}")
    
    return auctions, bidders, bids


def test_feature_engineering(auctions, bidders, bids):
    """Test feature engineering"""
    print("\n" + "="*80)
    print("TEST 2: Feature Engineering")
    print("="*80)
    
    engineer = FeatureEngineer(auctions, bidders, bids)
    features = engineer.combine_all_features()
    
    print(f"✓ Extracted {len(features.columns) - 1} features")
    print(f"✓ Features computed for {len(features)} bidders")
    
    feature_names = engineer.get_feature_names()
    print(f"\nFeatures: {feature_names[:5]} ... (showing first 5)")
    
    return features


def test_anomaly_detection(features):
    """Test anomaly detection"""
    print("\n" + "="*80)
    print("TEST 3: Anomaly Detection")
    print("="*80)
    
    detector = AnomalyDetector(features)
    predictions, scores = detector.detect_ensemble()
    
    anomaly_count = (predictions == -1).sum()
    print(f"✓ Detected {anomaly_count} anomalies")
    print(f"✓ Mean anomaly score: {scores.mean():.3f}")
    
    summary = detector.get_anomaly_summary()
    print(f"\nAnomaly Summary: {summary}")
    
    anomaly_df = detector.get_anomaly_dataframe()
    return anomaly_df


def test_network_analysis(bids):
    """Test network analysis"""
    print("\n" + "="*80)
    print("TEST 4: Network Analysis")
    print("="*80)
    
    analyzer = BidderNetworkAnalysis(bids)
    analyzer.build_cobidding_network()
    analyzer.detect_communities()
    
    stats = analyzer.get_network_statistics()
    print(f"✓ Network nodes: {stats['num_nodes']}")
    print(f"✓ Network edges: {stats['num_edges']}")
    print(f"✓ Network density: {stats['density']:.3f}")
    
    suspicious_pairs = analyzer.find_suspicious_pairs()
    print(f"✓ Found {len(suspicious_pairs)} suspicious pairs")
    
    if suspicious_pairs:
        print(f"  Top pair: {suspicious_pairs[0]}")
    
    return analyzer


def test_risk_scoring(features, anomaly_df, network_analyzer):
    """Test risk scoring"""
    print("\n" + "="*80)
    print("TEST 5: Risk Scoring")
    print("="*80)
    
    # Get network scores
    network_scores = {}
    for bidder_id in features['bidder_id']:
        network_scores[bidder_id] = network_analyzer.get_bidder_network_score(bidder_id)
    
    # Score risks
    scorer = CollusionRiskScorer(features, anomaly_df, network_scores)
    risk_scores = scorer.calculate_all_risk_scores()
    
    summary = scorer.get_scoring_summary()
    print(f"✓ High risk bidders: {summary['high_risk_count']}")
    print(f"✓ Medium risk bidders: {summary['medium_risk_count']}")
    print(f"✓ Mean risk score: {summary['mean_risk_score']:.3f}")
    
    high_risk = risk_scores[risk_scores['risk_level'] == 'HIGH']
    if len(high_risk) > 0:
        print(f"\nTop high-risk bidder: {high_risk.iloc[0]['bidder_id']} (score: {high_risk.iloc[0]['combined_risk_score']:.3f})")
    
    return risk_scores, scorer


def test_full_pipeline():
    """Test complete pipeline"""
    print("\n" + "="*80)
    print("TEST 6: Full Pipeline")
    print("="*80)
    
    config = load_config(preset_name='balanced')
    pipeline = CollusionDetectionPipeline(config=config, output_dir='./test_results')
    
    results = pipeline.run_pipeline()
    print("✓ Pipeline completed successfully")
    
    return results


def run_all_tests():
    """Run all tests"""
    print("\n\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "BIDDER COLLUSION DETECTION SYSTEM - TEST SUITE".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    try:
        # Test individual components
        auctions, bidders, bids = test_data_generation()
        features = test_feature_engineering(auctions, bidders, bids)
        anomaly_df = test_anomaly_detection(features)
        network_analyzer = test_network_analysis(bids)
        risk_scores, scorer = test_risk_scoring(features, anomaly_df, network_analyzer)
        
        # Test full pipeline
        results = test_full_pipeline()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
