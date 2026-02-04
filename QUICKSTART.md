# QUICKSTART Guide (5 minutes)

## Installation

```bash
# 1. Install Python 3.7+
python --version  # Should be 3.7 or higher

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python -c "import pandas, sklearn, networkx; print('✓ All packages installed')"
```

## Running the System

### Option 1: Quick Test (2 minutes)
```bash
python test_demo.py
```
This runs all components on a small dataset and shows you what the system can do.

### Option 2: Full Pipeline (5 minutes)
```bash
python main_pipeline.py
```
This generates data, runs full analysis, and saves results to `results/` directory.

### Option 3: Custom Analysis
```python
from main_pipeline import CollusionDetectionPipeline
from config import load_config

# Load a configuration preset
config = load_config(preset_name='aggressive')  # or 'balanced', 'conservative', etc.

# Create and run pipeline
pipeline = CollusionDetectionPipeline(config=config)
results = pipeline.run_pipeline()

# Access results
print(results['risk_scores'])  # Risk scores for all bidders
```

## Configuration Presets

| Preset | Use Case | Precision | Recall |
|--------|----------|-----------|--------|
| `conservative` | Minimize false positives | High | Low |
| `balanced` | Default, good balance | Medium | Medium |
| `aggressive` | Detect all possible cartels | Low | High |
| `network_focused` | Emphasize network patterns | - | - |
| `behavior_focused` | Focus on bidding behavior | - | - |
| `large_scale` | For 1000+ auctions | - | - |

**Usage:**
```python
config = load_config(preset_name='aggressive')
pipeline = CollusionDetectionPipeline(config=config)
pipeline.run_pipeline()
```

## Output Files

After running, check `results/` for:

- `risk_scores_*.csv` - **Main output**: Collusion risk scores for each bidder
- `anomaly_scores_*.csv` - Individual anomaly detection scores
- `features_*.csv` - Extracted features
- `risk_distribution.png` - Risk score histogram
- `network_graph.png` - Bidder network visualization
- Other PNG files for detailed analysis

## Interpreting Results

### Risk Scores
- **0.0 - 0.3**: Very Low Risk
- **0.3 - 0.6**: Low Risk
- **0.6 - 0.8**: Medium Risk
- **0.8 - 1.0**: High Risk ⚠️

### Columns in risk_scores.csv
- `bidder_id`: Bidder identifier
- `anomaly_score`: Unusual bidding patterns (0-1)
- `network_score`: Co-bidding with suspicious groups (0-1)
- `behavior_score`: Suspicious behavioral patterns (0-1)
- `combined_risk_score`: Overall collusion likelihood (0-1)
- `risk_level`: Categorical risk level (VERY_LOW, LOW, MEDIUM, HIGH)

## Common Tasks

### Find high-risk bidders
```python
import pandas as pd
scores = pd.read_csv('results/risk_scores_*.csv')
high_risk = scores[scores['risk_level'] == 'HIGH']
print(high_risk)
```

### Change number of auctions/bidders
```python
from config import DetectionConfig
config = DetectionConfig()
config.num_auctions = 500  # More auctions
config.num_bidders = 200   # More bidders
```

### Adjust risk thresholds
```python
config = DetectionConfig()
config.risk_threshold_high = 0.75  # Lower threshold = more flagged bidders
```

### Use only specific algorithms
```python
from anomaly_detection import AnomalyDetector
detector = AnomalyDetector(features)

# Isolation Forest only
predictions, scores = detector.detect_isolation_forest()

# LOF only
predictions, scores = detector.detect_lof()
```

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'pandas'`
**Solution:** Run `pip install -r requirements.txt`

### Issue: `community` module not found
**Solution:** Run `pip install python-louvain`

### Issue: Out of memory with large dataset
**Solution:** 
1. Reduce `num_auctions` or `num_bidders` in config
2. Use `large_scale` preset: `load_config(preset_name='large_scale')`

### Issue: Results folder not created
**Solution:** Manually create `results/` folder: `mkdir results`

### Issue: Plots not saving
**Solution:** Ensure `matplotlib` is installed: `pip install matplotlib`

## Next Steps

1. ✅ Run `python test_demo.py`
2. ✅ Run `python main_pipeline.py`
3. ✅ Check results in `results/` directory
4. ✅ Read [README.md](README.md) for advanced options
5. ✅ Customize configuration as needed

---

**Questions?** See [README.md](README.md) for comprehensive documentation.
