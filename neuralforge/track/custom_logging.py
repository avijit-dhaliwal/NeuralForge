# neuralforge/track/custom_logging.py
import mlflow
import matplotlib.pyplot as plt
import io

class CustomLogger:
    def __init__(self, experiment_name):
        mlflow.set_experiment(experiment_name)

    def log_metric(self, key, value, step=None):
        mlflow.log_metric(key, value, step=step)

    def log_custom_chart(self, figure, artifact_path):
        buf = io.BytesIO()
        figure.savefig(buf, format='png')
        buf.seek(0)
        mlflow.log_figure(figure, artifact_path)

    def log_confusion_matrix(self, cm, class_names):
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names, yticklabels=class_names,
               title='Confusion matrix',
               ylabel='True label',
               xlabel='Predicted label')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2. else "black")
        self.log_custom_chart(fig, "confusion_matrix.png")

# Usage:
# logger = CustomLogger("MyExperiment")
# with mlflow.start_run():
#     logger.log_metric("accuracy", 0.95)
#     logger.log_confusion_matrix(confusion_matrix, class_names)