"""
Jimmy Guo and David Xu
CSE 163 Final

Implements methods that generate two graphs:
one is a scatter plot of sugar percent vs
win percent, other is a scatter plot
of price percent vs win percent. We do this
to find the optimal sugar percent and
win percent that would maximize the win percent of a candy
for CSE 163 Final using the candy-data.csv dataset
"""
import pandas as pd
import plotly.express as px


def main():
    """
    Runs methods that generate graphs to help determine
    the sugar percent and price percent value that
    maximizes win percent of a candy.
    Prints out the optimal sugar percent and win percent
    """
    data = pd.read_csv('candy-data.csv')
    data = data.dropna()
    data = data[['winpercent', 'sugarpercent', 'pricepercent']]
    print("Optimal Sugar Percent: " + str(sugar_vs_win(data)))
    print("Optimal Price Percent: " + str(price_vs_win(data)))


def sugar_vs_win(data):
    """
    Returns the optimal sugar percent that maximizes
    win percent from trendline.
    Plots sugar percentage vs win percent of every candy.
    Takes dataset as parameter.
    """
    fig = px.scatter(data, x='sugarpercent', y='winpercent',
                     trendline="lowess")
    fig.update_layout(title="Sugar Percent vs Win Percent" +
                      " relative to other Candies")
    fig.show()
    return max(fig.data[1].y)


def price_vs_win(data):
    """
    Returns the optimal price percent that maximizes
    win percent from trendline.
    Plots price percentage vs win percent of every candy.
    Takes dataset as parameter.
    """
    fig2 = px.scatter(data, x='pricepercent', y='winpercent',
                      trendline="lowess")
    fig2.update_layout(title="Price Percent vs Win Percent" +
                       " relative to other Candies")
    fig2.show()
    return max(fig2.data[1].y)


if __name__ == "__main__":
    main()
