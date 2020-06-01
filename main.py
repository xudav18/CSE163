"""
Jimmy Guo and David Xu
CSE 163 Final

Implements methods to generate data visualizations for CSE 163 Final
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
	sns.set()
	data = pd.read_csv('candy-data.csv')
	data = data.dropna()
	data = data.set_index('competitorname')
	question_one(data)
	question_two(data)
	#question_three()

def question_one(data):
	features = data.loc[:, (data.columns != 'winpercent')]
	labels = data['winpercent']
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
	ax.set_xticklabels(x_axis, size = 7)
	ax.legend(fontsize = 8 )
	plt.xlabel("Competitor Name")
	plt.ylabel("Win Percentage")
	plt.title("Model Predictions vs Actual Win Percentage on Test Set")
	plt.xticks(rotation=90)
	plt.subplots_adjust(bottom=0.33)
	plt.savefig('model_predictions_vs_actual.png')
	print(mean_squared_error(labels_test, model.predict(features_test)))


if __name__ == "__main__":
	main()