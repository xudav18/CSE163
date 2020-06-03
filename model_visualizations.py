"""
Jimmy Guo and David Xu
CSE 163 Final

Implements methods to generate data visualizations
of ML Regressor model predictions vs actual
win percentages of candy for CSE 163 Final
using the candy-data.csv dataset
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error


def main():
    """
    Runs methods that generate graphs for model predictions
    vs actual win percentages based off every
    attribute as a whole and individually
    """
    sns.set()
    data = pd.read_csv('candy-data.csv')
    data = data.dropna()
    data = data.set_index('competitorname')
    question_one(data)
    question_two(data)


def question_one(data):
    """
    Plots predicted from model vs actual win percentage based off all
    candy attribute as a whole. Takes dataset as parameter.
    """
    features = data.loc[:, (data.columns != 'winpercent')]
    labels = data['winpercent']
    mean_sq_err = plot_pred_vs_actual(features, labels)
    print('Total: ' + str(mean_sq_err))
    plt.title("Model Predictions based on All Attributes " +
              "vs Actual Win Percentage")
    plt.savefig('model_predictions_vs_actual.png')


def question_two(data):
    """
    Plots predicted from model vs actual win percentage based off each
    candy attribute individually. Takes dataset as parameter.
    """
    features = data.loc[:, (data.columns != 'winpercent')]
    labels = data['winpercent']
    for feature in features.columns:
        mean_sq_err = plot_pred_vs_actual(
                      data[feature].values.reshape(-1, 1), labels)
        print(feature + ": " + str(mean_sq_err))
        plt.title("Model Predictions based off " + feature +
                  " vs Actual Win Percentage")
        plt.savefig(feature + '_model_predictions_vs_actual.png')


def plot_pred_vs_actual(features, labels):
    """
    Returns mean squared error for model predictions vs
    actual win percentage on test set.
    Helper method to generate plots for ML models based on a
    set of features and labels. Takes a dataframe features
    and dataframe labels as input.
    """
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.25)
    x_axis = labels_test.index
    x = np.arange(len(x_axis))
    width = 0.35
    model = DecisionTreeRegressor()
    model.fit(features_train, labels_train)
    predictions = model.predict(features_test)
    fig, ax = plt.subplots()
    ax.bar(x-width/2, predictions, width, label='Predictions')
    ax.bar(x + width/2, list(labels_test), width, label='Actual')
    ax.set_xticks(x)
    ax.set_xticklabels(x_axis, size=7)
    ax.legend(fontsize=8)
    plt.xlabel("Competitor Name")
    plt.ylabel("Win Percentage")
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.35)
    return mean_squared_error(labels_test, model.predict(features_test))


if __name__ == "__main__":
    main()
