# neuralforge/dashboard/app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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

st.title("NeuralForge Dashboard")

experiment_name = st.sidebar.text_input("Enter Experiment Name")
if experiment_name:
    df = load_experiment_data(experiment_name)
    st.write(f"Showing results for experiment: {experiment_name}")

    st.subheader("Runs Overview")
    st.dataframe(df[["run_id", "start_time"]])

    st.subheader("Metrics Visualization")
    metric = st.selectbox("Select Metric", list(df["metrics"].iloc[0].keys()))
    fig, ax = plt.subplots()
    ax.plot(df["start_time"], df["metrics"].apply(lambda x: x[metric]))
    ax.set_xlabel("Time")
    ax.set_ylabel(metric)
    st.pyplot(fig)

    st.subheader("Parameter Comparison")
    params = list(df["params"].iloc[0].keys())
    selected_params = st.multiselect("Select Parameters", params)
    if selected_params:
        param_df = pd.DataFrame({param: df["params"].apply(lambda x: x[param]) for param in selected_params})
        st.dataframe(param_df)

# Run with: streamlit run neuralforge/dashboard/app.py