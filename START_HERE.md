# START HERE 🚀

Welcome to the **Bidder Collusion and Cartel Detection System** for coal e-auctions! This is your entry point to a comprehensive machine learning system designed to detect suspicious bidding patterns and cartels.

## Quick Links

- **New to the project?** → Read [QUICKSTART.md](QUICKSTART.md) (5 minutes)
- **Want details?** → Read [README.md](README.md) (complete guide)
- **Need architecture info?** → See [ARCHITECTURE.md](ARCHITECTURE.md)
- **Looking for files?** → Check [FILE_INDEX.md](FILE_INDEX.md)

## 30-Second Overview

This system analyzes coal auction data to detect bidder collusion using:
- **Anomaly Detection** (Isolation Forest + Local Outlier Factor)
- **Network Analysis** (Co-bidding patterns, community detection)
- **Behavioral Features** (20+ features capturing bidding patterns)
- **Risk Scoring** (Combines all signals into collusion risk scores)
- **Visualizations** (7 chart types for insights)

## Immediate Next Steps

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the demo:**
   ```bash
   python test_demo.py
   ```

3. **Run the full pipeline:**
   ```bash
   python main_pipeline.py
   ```

4. **Check the results:**
   - Look in `results/` directory for outputs
   - CSV files contain detailed scores
   - PNG files contain visualizations

## What You Get

✅ Synthetic auction data with realistic collusion patterns  
✅ 20+ engineered features capturing bidding behavior  
✅ Ensemble anomaly detection combining multiple algorithms  
✅ Network analysis identifying suspicious groups  
✅ Comprehensive risk scoring (0-1 scale)  
✅ 7 publication-quality visualizations  
✅ Detailed logging of the entire analysis  

## Key Files

- `main_pipeline.py` - Complete analysis workflow
- `test_demo.py` - Quick test suite
- `config.py` - Configuration with 6 presets
- `risk_scores.csv` - Main output file with collusion risk scores

## Support

- For configuration options: See [config.py](config.py)
- For troubleshooting: See QUICKSTART.md "Troubleshooting" section
- For detailed documentation: See README.md

---

**Ready?** → Go to [QUICKSTART.md](QUICKSTART.md)
