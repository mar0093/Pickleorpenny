from datetime import datetime, time

import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
from matplotlib import ticker
import mpl_finance
import MySQLdb

def isolated_candle_graph():
    conn = MySQLdb.connect(host='localhost',
                           user='root',
                           passwd='pickle',#
                           db='pickledb') # falsified information
    cursor = conn.cursor()
    sql = "SELECT * FROM ddr"
    cursor.execute(sql)
    result = cursor.fetchall()
    print(result)
    df = pd.DataFrame(list(result), columns=["Date", "Open", "High", "Low", "Close", "Adj_Close", "Volume"])
    fig, ax = plt.subplots()

    mpl_finance.candlestick2_ohlc(ax, df.Open, df.High, df.Low, df.Close,
                      width=0.6, colorup='r', colordown='c', alpha=1)
    df = df.set_index(df["Date"])
    xdate = df.index

    def mydate(x, pos):
        try:
            return xdate[int(x)]
        except IndexError:
            return ''

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(mydate))
    plt.show()


isolated_candle_graph()

