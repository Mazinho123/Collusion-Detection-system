"""
Streamlit Frontend for Anomaly Detection System
Test and visualize collusive bidding detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from anomaly_detection import AnomalyDetector
import io

st.set_page_config(page_title="Collusion Detection", layout="wide")

st.title("🎯 Collusive Bidding Detection System")
st.markdown("Detect anomalous bidding behavior using ensemble machine learning methods")

# Sidebar for configuration
st.sidebar.header("Configuration")
contamination = st.sidebar.slider("Contamination Rate", 0.01, 0.5, 0.1, 0.01)
n_neighbors = st.sidebar.slider("LOF Neighbors", 5, 50, 20, 1)
ensemble_method = st.sidebar.radio("Ensemble Method", ["voting", "averaging"])

# Data input section
st.header("📊 Data Input")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Option 1: Upload CSV")
    uploaded_file = st.file_uploader("Upload bidding data (CSV)", type=['csv'])

with col2:
    st.subheader("Option 2: Generate Sample Data")
    num_bidders = st.number_input("Number of Bidders", 20, 1000, 100, 10)
    if st.button("Generate Sample Data"):
        st.session_state.use_sample = True
        st.session_state.num_bidders = num_bidders

# Load data
data = None
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success(f"✓ Loaded {len(data)} bidders")
    except Exception as e:
        st.error(f"Error loading file: {e}")

elif st.session_state.get("use_sample", False):
    # Generate truly random data each time (no fixed seed)
    num_bidders = st.session_state.num_bidders
    
    # Create realistic bidding data with randomization
    data = pd.DataFrame({
        'bidder_id': [f'bidder_{i:04d}' for i in range(num_bidders)],
        'bid_frequency': np.random.exponential(scale=30, size=num_bidders),
        'bid_variance': np.random.gamma(shape=2, scale=50, size=num_bidders),
        'win_rate': np.random.beta(a=2, b=5, size=num_bidders),
        'avg_bid_amount': np.random.lognormal(mean=8, sigma=1, size=num_bidders),
        'bid_pattern_regularity': np.random.uniform(0, 1, size=num_bidders),
        'price_sensitivity': np.random.normal(loc=0.5, scale=0.2, size=num_bidders)
    })
    
    # Generate purely random data - NO synthetic anomalies injected
    # Let the anomaly detection algorithm find natural anomalies
    
    st.success(f"✓ Generated {len(data)} bidders with RANDOM data - anomalies will be detected naturally")

# Display data preview
if data is not None:
    st.subheader("Data Preview")
    st.dataframe(data.head(10), use_container_width=True)
    st.write(f"Dataset shape: {data.shape}")
    
    # Run anomaly detection
    st.header("🔍 Anomaly Detection Analysis")
    
    try:
        detector = AnomalyDetector(data, contamination=contamination)
        
        # Get results
        result_df = detector.get_anomaly_dataframe()
        summary = detector.get_anomaly_summary()
        importance = detector.get_feature_importance(n_features=10)
        
        # Calculate normal ranges (from non-anomalous bidders)
        normal_bidders = result_df[result_df['is_anomaly'] == False]
        normal_ranges = {}
        for col in data.columns:
            if col != 'bidder_id':
                normal_ranges[col] = {
                    'min': normal_bidders[col].min() if col in normal_bidders.columns else data[col].min(),
                    'max': normal_bidders[col].max() if col in normal_bidders.columns else data[col].max(),
                    'mean': normal_bidders[col].mean() if col in normal_bidders.columns else data[col].mean(),
                    'std': normal_bidders[col].std() if col in normal_bidders.columns else data[col].std()
                }
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Bidders", int(summary['total_bidders']))
        with col2:
            anomaly_count = int(summary['num_anomalies'])
            st.metric("🚨 Anomalies Detected", anomaly_count, delta=f"{summary['anomaly_percentage']:.1f}%")
        with col3:
            st.metric("Normal Bidders", int(summary['total_bidders'] - anomaly_count))
        with col4:
            st.metric("Mean Anomaly Score", f"{summary['mean_anomaly_score']:.4f}")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Scores", "Anomalies", "Normal Ranges", "Importance", "Details", "Download"])
        
        with tab1:
            st.subheader("Anomaly Score Distribution")
            fig = px.histogram(result_df, x='ensemble_score', nbins=30, 
                             title="Distribution of Ensemble Anomaly Scores",
                             labels={'ensemble_score': 'Ensemble Score'},
                             color='is_anomaly',
                             color_discrete_map={True: '#FF6B6B', False: '#51CF66'})
            fig.add_vline(x=0.5, line_dash="dash", line_color="orange", 
                         annotation_text="Threshold (0.5)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Score comparison
            fig2 = go.Figure()
            fig2.add_trace(go.Box(y=result_df['isolation_forest_score'], 
                                 name='Isolation Forest', boxmean='sd'))
            fig2.add_trace(go.Box(y=result_df['lof_score'], 
                                 name='LOF', boxmean='sd'))
            fig2.add_trace(go.Box(y=result_df['ensemble_score'], 
                                 name='Ensemble', boxmean='sd'))
            fig2.update_layout(title="Score Comparison Across Methods", 
                             yaxis_title="Anomaly Score")
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            st.subheader("🚨 Detected Anomalies (Color-Coded)")
            
            if len(result_df[result_df['is_anomaly'] == True]) > 0:
                # Display anomalies with highlighting
                anomalies = result_df[result_df['is_anomaly'] == True].head(20).copy()
                
                # Create a color-coded display
                st.markdown("""
                <style>
                .anomaly-row {
                    background-color: #ffcccc;
                    padding: 10px;
                    margin: 5px 0;
                    border-left: 5px solid #ff0000;
                    border-radius: 5px;
                }
                .normal-row {
                    background-color: #ccffcc;
                    padding: 10px;
                    margin: 5px 0;
                    border-left: 5px solid #00cc00;
                    border-radius: 5px;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Display each anomaly with details
                for idx, row in anomalies.iterrows():
                    with st.container():
                        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                        with col1:
                            st.markdown(f"<div class='anomaly-row'><b>🚨 {row['bidder_id']}</b></div>", 
                                      unsafe_allow_html=True)
                        with col2:
                            st.metric("Ensemble Score", f"{row['ensemble_score']:.3f}", 
                                    delta="HIGH RISK" if row['ensemble_score'] > 0.75 else "MEDIUM RISK")
                        with col3:
                            st.metric("IF Score", f"{row['isolation_forest_score']:.3f}")
                        with col4:
                            st.metric("LOF Score", f"{row['lof_score']:.3f}")
                        
                        # Show why it's anomalous
                        bidder_data = data[data['bidder_id'] == row['bidder_id']].iloc[0] if len(data[data['bidder_id'] == row['bidder_id']]) > 0 else None
                        if bidder_data is not None:
                            st.write("**Why flagged as anomaly?**")
                            reasons = []
                            for col in data.columns:
                                if col != 'bidder_id' and col in normal_ranges:
                                    val = bidder_data[col]
                                    range_info = normal_ranges[col]
                                    if val < range_info['min'] - range_info['std'] or val > range_info['max'] + range_info['std']:
                                        reasons.append(f"• {col}: {val:.2f} (Normal range: {range_info['min']:.2f} - {range_info['max']:.2f})")
                            if reasons:
                                st.write("\n".join(reasons))
                        st.divider()
                
                st.write(f"Showing {min(20, len(anomalies))} of {len(result_df[result_df['is_anomaly'] == True])} anomalies")
            else:
                st.success("✓ No anomalies detected! All bidders appear normal.")
        
        with tab3:
            st.subheader("✓ Normal Bidding Ranges")
            st.markdown("""
            The ranges below represent **normal, non-anomalous bidding behavior**. 
            Bidders should stay within these ranges to avoid being flagged as anomalous.
            """)
            
            # Create a table showing normal ranges
            ranges_data = []
            for feature in data.columns:
                if feature != 'bidder_id':
                    range_info = normal_ranges[feature]
                    ranges_data.append({
                        'Feature': feature,
                        'Minimum': f"{range_info['min']:.2f}",
                        'Maximum': f"{range_info['max']:.2f}",
                        'Mean': f"{range_info['mean']:.2f}",
                        'Std Dev': f"{range_info['std']:.2f}",
                        'Safe Range': f"{range_info['mean'] - range_info['std']:.2f} to {range_info['mean'] + range_info['std']:.2f}"
                    })
            
            ranges_df = pd.DataFrame(ranges_data)
            st.dataframe(ranges_df, use_container_width=True)
            
            st.markdown("""
            ### Interpretation:
            - **Safe Range**: Mean ± 1 Standard Deviation (68% of normal bidders fall here)
            - **Caution Zone**: Values outside 1-2 standard deviations (suspicious)
            - **Red Flag Zone**: Values outside 2+ standard deviations (highly anomalous)
            """)
            
            # Visualization of ranges
            st.subheader("Visual Range Comparison")
            
            for feature in list(data.columns)[:5]:  # Show first 5 features
                if feature != 'bidder_id':
                    range_info = normal_ranges[feature]
                    
                    fig = go.Figure()
                    
                    # Add range bands
                    fig.add_vrect(
                        x0=range_info['mean'] - range_info['std'],
                        x1=range_info['mean'] + range_info['std'],
                        fillcolor="green", opacity=0.2, layer="below",
                        annotation_text="Safe", annotation_position="top left"
                    )
                    
                    fig.add_vrect(
                        x0=range_info['min'],
                        x1=range_info['mean'] - range_info['std'],
                        fillcolor="orange", opacity=0.2, layer="below"
                    )
                    
                    fig.add_vrect(
                        x0=range_info['mean'] + range_info['std'],
                        x1=range_info['max'],
                        fillcolor="red", opacity=0.2, layer="below"
                    )
                    
                    # Add data points
                    colors = ['red' if result_df.iloc[i]['is_anomaly'] else 'blue' 
                             for i in range(len(data))]
                    fig.add_scatter(
                        x=data[feature],
                        y=[1]*len(data),
                        mode='markers',
                        marker=dict(size=8, color=colors),
                        name='Bidders'
                    )
                    
                    fig.update_layout(
                        title=f"Range Distribution: {feature}",
                        xaxis_title=feature,
                        height=300,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame(
                list(importance.items()),
                columns=['Feature', 'Importance Score']
            )
            
            fig = px.bar(importance_df, x='Importance Score', y='Feature',
                        orientation='h', title="Top Features for Anomaly Detection",
                        color='Importance Score',
                        color_continuous_scale='Reds')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(importance_df, use_container_width=True)
        
        with tab5:
            st.subheader("Detailed Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Summary Statistics**")
                summary_display = {
                    'Total Bidders': int(summary['total_bidders']),
                    'Anomalies': int(summary['num_anomalies']),
                    'Percentage': f"{summary['anomaly_percentage']:.2f}%",
                    'Mean Score': f"{summary['mean_anomaly_score']:.6f}",
                    'Max Score': f"{summary['max_anomaly_score']:.6f}",
                    'Min Score': f"{summary['min_anomaly_score']:.6f}",
                    'Std Dev': f"{summary['std_anomaly_score']:.6f}"
                }
                for key, val in summary_display.items():
                    st.write(f"• {key}: {val}")
            
            with col2:
                st.write("**Configuration**")
                st.write(f"• Contamination Rate: {contamination}")
                st.write(f"• LOF Neighbors: {n_neighbors}")
                st.write(f"• Ensemble Method: {ensemble_method}")
                st.write(f"• Feature Count: {len(detector.feature_columns)}")
                st.write(f"• Total Records: {len(result_df)}")
            
            st.markdown("### Score Interpretation")
            st.markdown("""
            | Score Range | Classification | Meaning |
            |-------------|-----------------|---------|
            | 0.0 - 0.25 | ✓ Normal | Healthy bidder |
            | 0.25 - 0.50 | ⚠ Low Risk | Mostly normal |
            | 0.50 - 0.75 | ⚠ Medium Risk | Suspicious, review |
            | 0.75 - 1.00 | 🚨 High Risk | Likely collusive |
            """)
        
        with tab6:
            st.subheader("Download Results")
            
            # Prepare downloadable data with flagging
            export_df = result_df.copy()
            export_df['status'] = export_df['is_anomaly'].apply(lambda x: 'ANOMALY' if x else 'NORMAL')
            
            # Download full results
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download Full Results (CSV)",
                data=csv,
                file_name="anomaly_detection_results.csv",
                mime="text/csv"
            )
            
            # Download anomalies only
            anomalies_df = export_df[export_df['is_anomaly'] == True]
            csv_anomalies = anomalies_df.to_csv(index=False)
            st.download_button(
                label="Download Anomalies Only (CSV)",
                data=csv_anomalies,
                file_name="detected_anomalies.csv",
                mime="text/csv"
            )
            
            # Download normal ranges for reference
            csv_ranges = ranges_df.to_csv(index=False)
            st.download_button(
                label="Download Normal Ranges (CSV)",
                data=csv_ranges,
                file_name="normal_bidding_ranges.csv",
                mime="text/csv"
            )
            
            st.write("✓ Results ready for download")
        
        st.success("✓ Analysis completed successfully!")
        
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        st.write("Please check your data and try again")

else:
    st.info("👈 Please upload a CSV file or generate sample data to begin analysis")

# Footer
st.divider()
st.markdown("""
### System Status: ✓ Operational
- Anomaly Detection Engine: Active
- Feature Processing: Ready
- Model Training: Enabled
- Export Functionality: Available
- Normal Range Calculation: Available
""")

