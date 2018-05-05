import datetime
import matplotlib.pyplot as plt
from PIL import Image


class Plot(object):
    def __init__(self):
        ''' Constructor '''

    @staticmethod
    def drawTrend(data, coin):
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[5, 1]}, figsize=(10, 10))

        ax1.set_ylabel('Closing Price ($)', fontsize=12)
        ax1.set_xticks([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 7]])
        ax1.set_xticklabels('')


        ax2.set_ylabel('Volume ($ ' + coin + ')', fontsize=12)
        ax2.set_yticks([int('%d000000000' %i) for i in range(10)])
        ax2.set_yticklabels(range(10))
        ax2.set_xticks([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 7]])
        ax2.set_xticklabels([datetime.date(i, j, 1).strftime('%b %Y')  for i in range(2013, 2019) for j in [1, 7]])

        ax1.plot(data['Date'].astype(datetime.datetime), data['Open'])
        ax2.bar(data['Date'].astype(datetime.datetime).values, data['Volume'].values)
        fig.tight_layout()
        plt.show()
        fig.savefig("assets/" + coin + "Trend.png")

    @staticmethod
    def drawTest(data, split_date, target, coin):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.set_xticks([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 7]])
        ax.set_xticklabels([datetime.date(i, j, 1).strftime('%b %Y')  for i in range(2013, 2019) for j in [1, 7]])
        ax.plot(data[data['Date'] < split_date]['Date'].astype(datetime.datetime),
                data[data['Date'] < split_date][target],
                color='#B08FC7')
        ax.plot(data[data['Date'] >= split_date]['Date'].astype(datetime.datetime),
                data[data['Date'] >= split_date][target],
                color='#8FBAC8')
        ax.set_ylabel(coin + ' Price ($)', fontsize=12)
        plt.tight_layout()
        plt.show()
        fig.savefig("assets/" + coin + "Test.png")
