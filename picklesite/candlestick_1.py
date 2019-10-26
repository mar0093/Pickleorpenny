from datetime import datetime, time
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.dates as dates
import matplotlib.pyplot as plt
from matplotlib import ticker
import mpl_finance
import MySQLdb
import matplotlib.dates as mdates

def my_df():
    conn = MySQLdb.connect(host='localhost',
                           user='root',
                           passwd='pickle',  #
                           db='pickledb')  # falsified information
    cursor = conn.cursor()
    sql = "SELECT * FROM ddr"
    cursor.execute(sql)
    result = cursor.fetchall()
    df = pd.DataFrame(list(result), columns=["Date", "Open", "High", "Low", "Close", "Adj_Close", "Volume"])
    # fig, ax = plt.subplots()

    df["Date"] = [dt.datetime.strptime(d, '%Y-%m-%d').toordinal() for d in df["Date"]]
    df = df.set_index(df["Date"])
    # ax.plot_date(df.index, df["Volume"], fmt='o', tz=None, xdate=True)
    # plt.show()
    return df

def candle_stick(df):
    quotes = [tuple([df["Date"].tolist()[i],
                     df["Open"].tolist()[i],
                     df["High"].tolist()[i],
                     df["Low"].tolist()[i],
                     df["Close"].tolist()[i]]) for i in range(len(df.index))]  # _1

    fig, [ax, ax2] = plt.subplots(2,1,sharex=True)

    mpl_finance.candlestick_ohlc(ax, quotes, width=0.6)
    df["Close_av_100"] = df['Close'].rolling(100).mean()
    df["Close_av_30"] = df['Close'].rolling(30).mean()
    #pricing["Close_av_100"] = pd.rolling_mean(pricing["Close"],100)
    ax.plot(df["Date"].tolist(), df["Close_av_100"])
    ax.plot(df["Date"].tolist(), df["Close_av_30"])
    ax2.plot(df["Date"].tolist(), df["Volume"])
    fig.autofmt_xdate()
    fig.tight_layout()

    ax.xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d'))
    ax.grid(True)
    plt.show()

df = my_df()
candle_stick(df)