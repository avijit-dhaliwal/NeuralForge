# neuralforge/interpret/shap_explainer.py
import shap
import numpy as np
import matplotlib.pyplot as plt

class SHAPExplainer:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.explainer = shap.KernelExplainer(model.predict, data)

    def compute_shap_values(self, X):
        return self.explainer.shap_values(X)

    def plot_summary(self, shap_values, feature_names):
        shap.summary_plot(shap_values, self.data, feature_names=feature_names)

    def plot_dependence(self, shap_values, feature_names, feature_idx):
        shap.dependence_plot(feature_idx, shap_values, self.data, feature_names=feature_names)

# Usage example:
explainer = SHAPExplainer(my_model, X_train)
shap_values = explainer.compute_shap_values(X_test)
explainer.plot_summary(shap_values, feature_names)
explainer.plot_dependence(shap_values, feature_names, feature_idx=0)