# neuralforge/dashboard/interactive_viz.py
import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import mlflow

def load_experiment_data(experiment_name):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(experiment.experiment_id)
    data = []
    for run in runs:
        data.append({
            "run_id": run.info.run_id,
            "start_time": run.info.start_time,
            "metrics": run.data.metrics,
            "params": run.data.params
        })
    return pd.DataFrame(data)

def plot_metric_comparison(df, metric_name):
    fig = px.scatter(df, x='start_time', y=f'metrics.{metric_name}',
                     hover_data=['run_id', 'params'],
                     title=f'{metric_name} over time')
    return fig

def plot_param_importance(df, metric_name):
    param_cols = [col for col in df.columns if col.startswith('params.')]
    corr_data = []
    for param in param_cols:
        corr = df[f'metrics.{metric_name}'].corr(pd.to_numeric(df[param], errors='coerce'))
        corr_data.append({'param': param, 'correlation': corr})
    
    corr_df = pd.DataFrame(corr_data)
    fig = px.bar(corr_df, x='param', y='correlation',
                 title=f'Parameter importance for {metric_name}')
    return fig

st.title("NeuralForge Interactive Dashboard")

experiment_name = st.sidebar.text_input("Enter Experiment Name")
if experiment_name:
    df = load_experiment_data(experiment_name)
    
    st.subheader("Metric Comparison")
    metric_name = st.selectbox("Select Metric", list(df['metrics'].iloc[0].keys()))
    st.plotly_chart(plot_metric_comparison(df, metric_name))
    
    st.subheader("Parameter Importance")
    st.plotly_chart(plot_param_importance(df, metric_name))

# Usage: streamlit run neuralforge/dashboard/interactive_viz.py