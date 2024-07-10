import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Model names and metrics
models = ['Logistic Regression', 'Random Forest', 'SVM']
accuracies = [0.82, 0.81, 0.80]

# Precision, Recall, F1-Score for each model (Class 0 and Class 1)
metrics = {
    'Logistic Regression': {'precision': [0.86, 0.69], 'recall': [0.90, 0.60], 'f1-score': [0.88, 0.64]},
    'Random Forest': {'precision': [0.84, 0.67], 'recall': [0.91, 0.53], 'f1-score': [0.87, 0.59]},
    'SVM': {'precision': [0.84, 0.67], 'recall': [0.91, 0.51], 'f1-score': [0.87, 0.58]}
}

# Confusion matrices for each model
confusion_matrices = {
    'Logistic Regression': np.array([[934, 102], [150, 223]]),
    'Random Forest': np.array([[938, 98], [174, 199]]),
    'SVM': np.array([[942, 94], [181, 192]])
}

# Plotting accuracy comparison
plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'green', 'red'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.ylim(0, 1)
plt.show()

# Plotting confusion matrices
for model in models:
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrices[model], annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model} - Confusion Matrix')
    plt.show()

# Plotting precision, recall, and f1-score
metric_names = ['precision', 'recall', 'f1-score']
for metric in metric_names:
    plt.figure(figsize=(12, 6))
    bar_width = 0.25
    index = np.arange(2)  # Two classes: 0 and 1
    for i, model in enumerate(models):
        plt.bar(index + i * bar_width, metrics[model][metric], bar_width, label=model)

    plt.xlabel('Class')
    plt.ylabel(metric.capitalize())
    plt.title(f'Comparison of {metric.capitalize()}')
    plt.xticks(index + bar_width, ['No Churn', 'Churn'])
    plt.legend()
    plt.ylim(0, 1)
    plt.show()
