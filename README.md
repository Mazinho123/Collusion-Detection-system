# README - Bidder Collusion Detection System

## Overview

This is a comprehensive machine learning system for detecting bidder collusion and cartels in coal e-auctions. It combines multiple detection techniques:

- **Anomaly Detection**: Isolation Forest + Local Outlier Factor (LOF) ensemble
- **Network Analysis**: Co-bidding patterns and community detection
- **Feature Engineering**: 20+ behavioral features
- **Risk Scoring**: Weighted combination of all signals
- **Explainability**: Feature importance and component breakdown
- **Visualization**: 7 publication-quality charts

## Features

### Core Algorithms

#### 1. Anomaly Detection 
- **Isolation Forest**: Detects global outliers
  - 100 decision trees
  - Contamination parameter: 10% (configurable)
  - Scores represent deviation from normal patterns

- **Local Outlier Factor (LOF)**: Detects local density anomalies
  - 20 nearest neighbors (configurable)
  - Captures local clustering patterns
  - Better for local anomalies

- **Ensemble**: Combines both algorithms
  - Voting method for robustness
  - Normalizes scores to [0, 1]
  - Reduces false positives

#### 2. Network Analysis 
- **Co-bidding Network**: Graph where edges = co-participation in auctions
- **Community Detection**: Louvain algorithm to identify bidding cartels
- **Centrality Measures**:
  - Degree centrality (most connected)
  - Betweenness centrality (network bridges)
  - Closeness centrality (overall connectivity)

#### 3. Feature Engineering
Extracts 20+ features per bidder:

**Participation Features**:
- Number of auctions
- Total bids
- Participation rate
- Average bids per auction

**Price Features**:
- Average/median/std bid price
- Price coefficient of variation
- Win rate
- First bid ratio

**Temporal Features**:
- Bidding interval consistency
- Preferred bidding hour
- Bidding regularity

**Network Features**:
- Number of co-bidders
- Co-bidder concentration

#### 4. Risk Scoring 
Combines signals with configurable weights:
```
Combined Risk = 0.5 × Anomaly Score
              + 0.3 × Network Score
              + 0.2 × Behavior Score
```

### Risk Levels
- **VERY_LOW** (0.0 - 0.3): Normal bidding
- **LOW** (0.3 - 0.6): Minor concerns
- **MEDIUM** (0.6 - 0.8): Suspicious pattern
- **HIGH** (0.8 - 1.0): Likely collusion ⚠️

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup
```bash
# 1. Navigate to project directory
cd CollusionDetection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python test_demo.py
```

### Dependencies
- **pandas** (1.3.0+): Data manipulation
- **numpy** (1.21.0+): Numerical computing
- **scikit-learn** (1.0.0+): ML algorithms
- **networkx** (2.6.0+): Network analysis
- **matplotlib** (3.4.0+): Visualization
- **seaborn** (0.11.0+): Statistical visualization
- **scipy** (1.7.0+): Scientific computing
- **python-louvain** (0.15+): Community detection

## Usage

### Quick Start
```bash
# Run demo test suite
python test_demo.py

# Run full pipeline
python main_pipeline.py
```

### Python API
```python
from main_pipeline import CollusionDetectionPipeline
from config import load_config

# Load configuration
config = load_config(preset_name='balanced')

# Create pipeline
pipeline = CollusionDetectionPipeline(config=config)

# Run analysis
results = pipeline.run_pipeline()

# Access results
risk_scores = results['risk_scores']
print(risk_scores.head())
```

### Custom Configuration
```python
from config import DetectionConfig

config = DetectionConfig()
config.num_auctions = 500
config.num_bidders = 200
config.isolation_forest_contamination = 0.08
config.anomaly_score_weight = 0.6  # Weight anomalies more
config.risk_threshold_high = 0.75  # Lower threshold for more alerts

pipeline = CollusionDetectionPipeline(config=config)
pipeline.run_pipeline()
```

### Using Presets
```python
from config import load_config

# Available presets:
# 'conservative' - high precision, low false positives
# 'balanced' - default, good balance
# 'aggressive' - high recall, catches more
# 'network_focused' - emphasize network patterns
# 'behavior_focused' - emphasize behavioral patterns
# 'large_scale' - for large datasets

config = load_config(preset_name='aggressive')
pipeline = CollusionDetectionPipeline(config=config)
pipeline.run_pipeline()
```

## Output

### Files Generated
- `risk_scores_*.csv` - Main output with scores for each bidder
- `anomaly_scores_*.csv` - Anomaly detection results
- `features_*.csv` - Extracted features
- `auctions_*.csv`, `bidders_*.csv`, `bids_*.csv` - Input data
- `*.png` files - Visualizations

### Visualizations (7 types)
1. **Risk Distribution** - Histogram + pie chart of risk levels
2. **Score Components** - Bar chart showing anomaly/network/behavior breakdown
3. **Network Graph** - Top 30 bidders and co-bidding connections
4. **Anomaly Scores** - Top 30 bidders by anomaly score
5. **Feature Importance** - Top 5 features in detection
6. **Bidding Timeline** - Bid price evolution for top bidders
7. **Auction Competition** - Distribution of bids per auction

### Risk Scores Output
CSV with columns:
- `bidder_id`: Bidder identifier
- `anomaly_score`: 0-1, higher = more unusual
- `network_score`: 0-1, higher = suspicious network
- `behavior_score`: 0-1, higher = suspicious behavior
- `combined_risk_score`: 0-1, main metric
- `risk_level`: VERY_LOW, LOW, MEDIUM, or HIGH

## Pipeline Steps

```
Step 1: Data Generation
  └─ Generate synthetic auctions + bids with collusion patterns

Step 2: Feature Engineering
  └─ Extract 20+ features for each bidder

Step 3: Anomaly Detection
  ├─ Isolation Forest
  ├─ Local Outlier Factor
  └─ Ensemble combination

Step 4: Network Analysis
  ├─ Build co-bidding network
  ├─ Louvain community detection
  └─ Calculate centrality measures

Step 5: Risk Scoring
  ├─ Calculate behavior score
  ├─ Combine with anomaly + network
  └─ Assign risk level

Step 6: Visualization & Output
  └─ Generate charts and save results
```

## Configuration

### Key Parameters

**Data Generation**
- `num_auctions`: Number of auctions (default: 100)
- `num_bidders`: Number of bidders (default: 50)
- `num_collusion_groups`: Number of collusion groups (default: 3)
- `collusion_percentage`: % auctions with collusion (default: 0.25)

**Anomaly Detection**
- `isolation_forest_contamination`: Expected anomaly % (default: 0.1)
- `isolation_forest_n_estimators`: Number of trees (default: 100)
- `lof_n_neighbors`: Neighbors for LOF (default: 20)
- `lof_contamination`: Expected anomaly % (default: 0.1)

**Risk Scoring Weights**
- `anomaly_score_weight`: Weight for anomaly score (default: 0.5)
- `network_score_weight`: Weight for network score (default: 0.3)
- `behavior_score_weight`: Weight for behavior score (default: 0.2)

**Risk Thresholds**
- `risk_threshold_low`: High limit (default: 0.3)
- `risk_threshold_medium`: High limit (default: 0.6)
- `risk_threshold_high`: High limit (default: 0.8)

See `config.py` for all options.

## Advanced Usage

### Analyzing Specific Bidders
```python
scorer = pipeline.risk_scorer
explanation = scorer.explain_risk_score(bidder_id=5)
print(f"Bidder 5: {explanation}")
```

### Detecting Communities
```python
analyzer = pipeline.network_analyzer
communities = analyzer.detect_communities()
print(f"Found {len(communities)} communities")
```

### Extracting Feature Importance
```python
detector = pipeline.anomaly_detector
importance = detector.get_feature_importance(n_features=5)
print(f"Top features: {importance}")
```

### Getting Suspicious Pairs
```python
analyzer = pipeline.network_analyzer
suspicious = analyzer.find_suspicious_pairs()
for bidder1, bidder2, co_bids in suspicious[:5]:
    print(f"Bidders {bidder1} and {bidder2} co-bid {co_bids} times")
```

## Metrics

### Performance Metrics
- **Precision**: % of detected collusions that are true positives
- **Recall**: % of true collusions detected
- **F1-Score**: Harmonic mean of precision and recall

### Quality Metrics
- **AUC-ROC**: Area under receiver operating characteristic curve
- **Mean Reciprocal Rank**: Ranking quality of detections
- **Coverage**: % of bidders with scores

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named '...'`
- **Solution**: Run `pip install -r requirements.txt`

**Issue**: Memory error with large dataset
- **Solution**: Reduce `num_auctions`/`num_bidders` or use `large_scale` preset

**Issue**: No results generated
- **Solution**: Check `results/` directory exists, or create manually

**Issue**: Slow execution
- **Solution**: Reduce dataset size or increase `n_jobs` in config

## Examples

### Example 1: Quick Analysis
```python
from main_pipeline import CollusionDetectionPipeline

pipeline = CollusionDetectionPipeline()
results = pipeline.run_pipeline()
high_risk = results['risk_scores'][results['risk_scores']['risk_level'] == 'HIGH']
print(f"Found {len(high_risk)} high-risk bidders")
```

### Example 2: Finding Cartels
```python
analyzer = pipeline.network_analyzer
communities = analyzer.detect_communities()

for comm_id, members in communities.items():
    if len(members) >= 3:
        print(f"Potential cartel: Bidders {members}")
```

### Example 3: Aggressive Detection
```python
config = load_config(preset_name='aggressive')
pipeline = CollusionDetectionPipeline(config=config)
results = pipeline.run_pipeline()
```

## Literature & Algorithms

- **Isolation Forest**: Liu et al., "Isolation Forest" (2008)
- **Local Outlier Factor**: Breunig et al., "LOF: Identifying Density-Based Local Outliers" (2000)
- **Louvain Community Detection**: Blondel et al., "Fast unfolding of communities in large networks" (2008)

## References

- Coal auction collusion studies
- Antitrust economics research
- Machine learning detection methods
- Network analysis techniques

## Contact & Support

For issues or questions:
1. Check [QUICKSTART.md](QUICKSTART.md)
2. Review [ARCHITECTURE.md](ARCHITECTURE.md)
3. See [FILE_INDEX.md](FILE_INDEX.md) for file descriptions

## License

[Project License - if applicable]

## Acknowledgments

Based on research in auction economics, antitrust enforcement, and machine learning.

---

**Ready to start?** → Run `python test_demo.py`

