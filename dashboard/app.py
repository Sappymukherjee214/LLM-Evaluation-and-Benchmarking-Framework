import streamlit as st
import pandas as pd
import glob
import json
import os
import plotly.express as px
from metrics.fairness import FairnessEvaluator

st.set_page_config(page_title="LLM Research Framework", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stMetric { border: 1px solid #e0e0e0; padding: 15px; border-radius: 10px; background-color: white; }
    h1 { color: #1a1a1a; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ LLM Fairness & Performance Analytics")
st.markdown("---")

# Load experiments
experiment_files = glob.glob("experiments/*.json")

if not experiment_files:
    st.warning("No experiments found. Run evaluations first.")
else:
    # Sidebar: Experiment Selection
    st.sidebar.header("🔬 Experiment Selection")
    selected_files = st.sidebar.multiselect("Select Runs", options=experiment_files, default=experiment_files[-2:] if len(experiment_files) > 1 else experiment_files)

    data_summary = []
    all_results = {}

    for file in selected_files:
        with open(file, 'r') as f:
            exp = json.load(f)
            config = exp.get("config", {})
            perf = exp.get("performance", {})
            metrics = exp.get("summary_metrics", {})
            
            summary = {
                "ID": config.get("experiment_id"),
                "Model": config.get("model_id"),
                "Dataset": config.get("dataset_name"),
                "Throughput": perf.get("throughput_samples_per_sec", 0),
                "Runtime": perf.get("total_runtime_sec", 0)
            }
            summary.update(metrics)
            data_summary.append(summary)
            all_results[config.get("experiment_id")] = exp.get("results", [])

    df_summary = pd.DataFrame(data_summary)

    # 1. High Level Metrics
    cols = st.columns(len(df_summary))
    for i, (_, row) in enumerate(df_summary.iterrows()):
        with cols[i % len(cols)]:
            st.metric(f"{row['Model']} Accuracy", f"{row.get('accuracy', 0):.2%}")

    # 2. Performance & Scalability
    st.header("⚡ Performance & Scalability")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        fig_tp = px.bar(df_summary, x="Model", y="Throughput", color="Dataset", title="Throughput (Samples/sec)")
        st.plotly_chart(fig_tp, use_container_width=True)
    with col_p2:
        fig_rt = px.scatter(df_summary, x="Runtime", y="accuracy", color="Model", size="Throughput", title="Accuracy vs. Runtime Efficiency")
        st.plotly_chart(fig_rt, use_container_width=True)

    # 3. Fairness & Bias Evaluation
    st.header("⚖️ Fairness & Bias Analysis")
    # Identify protected attributes in metadata
    exp_to_analyze = st.selectbox("Select Experiment for Bias Check", options=list(all_results.keys()))
    results = all_results[exp_to_analyze]
    
    # Check for demographic metadata
    possible_attributes = ["gender", "category", "subject", "difficulty", "domain"]
    found_attributes = [a for a in possible_attributes if any(a in r.get("metadata", {}) for r in results)]
    
    if found_attributes:
        attr = st.selectbox("Protected Attribute", options=found_attributes)
        metric_to_check = st.selectbox("Metric for Disparity", options=[c for c in df_summary.columns if c not in ["ID", "Model", "Dataset", "Throughput", "Runtime"]])
        
        evaluator = FairnessEvaluator(attr)
        group_scores = evaluator.compute_group_metrics(results, metric_to_check)
        disparity = evaluator.calculate_disparity(group_scores)
        
        st.subheader(f"Group Fairness: {attr.capitalize()}")
        col_f1, col_f2 = st.columns([2, 1])
        
        with col_f1:
            fig_fair = px.bar(
                x=list(group_scores.keys()), 
                y=list(group_scores.values()),
                labels={"x": attr.capitalize(), "y": metric_to_check.capitalize()},
                title=f"{metric_to_check.capitalize()} across {attr.capitalize()}"
            )
            st.plotly_chart(fig_fair, use_container_width=True)
        
        with col_f2:
            st.metric("Group Disparity", f"{disparity:.4f}", delta="Lower is fairer", delta_color="inverse")
            st.info(f"A disparity of {disparity:.2f} indicates the performance gap between the best and worst performing {attr} group.")

    # 4. Error Analysis
    st.header("🔍 Error Analysis & Insights")
    all_errors = []
    for res in results:
        err = res.get("error_analysis", {}).get("type")
        if err:
            all_errors.append({"type": err, "sample_id": res["sample_id"]})
    
    if all_errors:
        df_errors = pd.DataFrame(all_errors)
        fig_err = px.pie(df_errors, names="type", title="Error Distribution")
        st.plotly_chart(fig_err)
    else:
        st.success("No significant errors detected in this subset.")

    # 5. Raw Data
    with st.expander("📄 View Raw Experiment Logs"):
        st.dataframe(df_summary, use_container_width=True)
