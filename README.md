# Customer Churn Prediction Model

This project provides a visual comparison of the performance of three machine learning models: Logistic Regression, Random Forest, and Support Vector Machine (SVM), specifically for customer churn prediction. The comparison includes accuracy, precision, recall, F1-score, and confusion matrices for each model.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Visualizations](#visualizations)
- [License](#license)

## Introduction

The script compares the performance of three models on a customer churn prediction task. The following metrics are visualized:

1. **Accuracy**
2. **Precision**
3. **Recall**
4. **F1-Score**
5. **Confusion Matrices**

## Installation

To run this project, ensure you have the following packages installed:

- matplotlib
- seaborn
- numpy

You can install the required packages using pip:

```sh
pip install matplotlib seaborn numpy
```

## Usage

1. Clone the repository:

```sh
git clone https://github.com/yourusername/churn-prediction-comparison.git
cd churn-prediction-comparison
```

2. Run the script:

```sh
python churn_comparison.py
```

## Visualizations

### Accuracy Comparison

A bar chart comparing the accuracy of Logistic Regression, Random Forest, and SVM models.

```python
plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'green', 'red'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.ylim(0, 1)
plt.show()
```

### Confusion Matrices

Heatmaps of the confusion matrices for each model.

```python
for model in models:
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrices[model], annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model} - Confusion Matrix')
    plt.show()
```

### Precision, Recall, and F1-Score

Bar charts comparing the precision, recall, and F1-score for each model, divided by class (No Churn and Churn).

```python
metric_names = ['precision', 'recall', 'f1-score']
for metric in metric_names:
    plt.figure(figsize=(12, 6))
    bar_width = 0.25
    index = np.arange(2)  # Two classes: No Churn and Churn
    for i, model in enumerate(models):
        plt.bar(index + i * bar_width, metrics[model][metric], bar_width, label=model)

    plt.xlabel('Class')
    plt.ylabel(metric.capitalize())
    plt.title(f'Comparison of {metric.capitalize()}')
    plt.xticks(index + bar_width, ['No Churn', 'Churn'])
    plt.legend()
    plt.ylim(0, 1)
    plt.show()
```
