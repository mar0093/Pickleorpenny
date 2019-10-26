import matplotlib.pyplot as plt
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression, Ridge
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import mysql.connector
import MySQLdb
import datetime as dt
from statistics import mean
import mpl_finance

def AI(stock,days):
    df = pd.read_csv('C:/Users/James/PycharmProjects/NAB/data_code/yahoo/{}.csv'.format(stock))
    df = df.dropna()
    use_len = df.shape[0]-1
    df = df[['Adj Close', 'Close', 'Open', 'High', 'Low', 'Volume']].tail(use_len)
    last_price = df['Adj Close'].iloc[-1]
    forecast_out = int(days) # predicting 30 days into future
    df['Prediction'] = df[['Adj Close']].shift(-forecast_out) #  label column with data shifted 30 units
    X = np.array(df.drop(['Prediction'], 1))
    X = preprocessing.scale(X)
    X_forecast = X[-forecast_out:] # set X_forecast equal to last 30
    X = X[:-forecast_out]
    y = np.array(df['Prediction']) # set to adj close
    y = y[:-forecast_out] # removing the last 30 days that are 'NaN'
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
    clf = Ridge(alpha=0)
    clf.fit(X_train, y_train)
    forecast_prediction = clf.predict(X_forecast)
    list_pred = list(forecast_prediction)
    cheese = df['Adj Close'].tolist()
    a_list = cheese + list_pred
    return a_list, last_price

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    """
    This is the modified Neural network

    It has a minmaxscaler for the traits (x values)
    The y values have been normalized
    """
    def __init__(self, x, y):
        np.random.seed(666)
        scaler = MinMaxScaler()
        self.input      = x
        self.train_mean = np.mean(x)
        self.train_std  = np.std(x)
        self.stdize     = (x - self.train_mean)/self.train_std
        self.z          = np.array([6.4])
        #self.z          = sum(y)
        self.weights1   = np.random.rand(self.input.shape[1], 27)
        self.weights2   = np.random.rand(27, 1)
        self.y          = [float(i)/self.z for i in y[:, ]]# normalize y data
        self.output     = np.zeros(y.shape)
        self.data_test  = self.y

    def denormalize_data(self, value):
        return float(value * self.z)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.stdize, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))


    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        #print("d_weights2 is " + str(d_weights2.shape))
        #d_weights1 = np.dot(np.squeeze(np.asarray(self.train_test.T)), (np.dot(np.squeeze(np.asarray(2 * (self.data_test - self.output) * sigmoid_derivative(self.output))), np.squeeze(np.asarray(self.weights2.T) * sigmoid_derivative(self.layer1)))))
        d_weights1 = np.dot(self.stdize.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1))) #original (broken for my case)

        #print("d_weights1 is " + str(d_weights1.shape))
        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        #print("self.weights1 is " + str(self.weights1.shape))

    # The neural network thinks.
    def think(self, inputs):
        std_value = (inputs - self.train_mean)/self.train_std
        self.layer1 = sigmoid(np.dot(std_value, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        return self.output

def aquire_df(aticker):
    db = MySQLdb.connect(host="localhost",  # your host, usually localhost
                         user="root",  # your username
                         passwd="pickle",  # your password
                         db="pickledb")  # name of the data base

    # you must create a Cursor object. It will let
    #  you execute all the queries you need
    cur = db.cursor()
    # Get articles
    result = cur.execute("SELECT * FROM " + aticker)
    # Get all articles
    my_result = cur.execute("SELECT * FROM " + aticker + " ORDER BY Date DESC LIMIT 1")
    my_result = cur.fetchone()
    sql = "SELECT * FROM " + aticker
    cur.execute(sql)
    all_result = cur.fetchall()
    df = pd.DataFrame(list(all_result), columns=["Date", "Open", "High", "Low", "Close", "Adj_Close", "Volume" ])

    #put at end
    db.close()
    return df

class Traits:
    """
    Have to remember to include a trait that brings open, close, high, low together.
    """
    def __init__(self, df, df_col):
        self.df = df
        self.df_col = df_col
        self.close_values = np.array(list(df["Close"].tail(120)), dtype=np.float64)
        self.open_values = np.array(list(df["Open"].tail(120)), dtype=np.float64)
        self.high_values = np.array(list(df["High"].tail(120)), dtype=np.float64)
        self.low_values = np.array(list(df["Low"].tail(120)), dtype=np.float64)

    def day_gradient(self, start, remove_days=0):
        if self.df_col == "Close":
            if remove_days == 0:
                ys = self.close_values[-start:, ]
            else:
                ys = self.close_values[-start:, ]
                ys = ys[:start-remove_days, ]
        x = list(range(0, start-remove_days))
        xs = np.array(x, dtype=np.float64)
        print(ys)
        print(xs)
        m = round((((mean(xs)*mean(ys)) - mean(xs*ys)) /
             ((mean(xs)**2) - mean(xs**2)))*100, 5)
        plt.show()
        print(m)
        return m, ys

    def sos_error(self, data_points):
        ys = data_points
        mean_ys = mean(ys)
        sum = 0
        for i in range(0,len(ys)):
            err = (ys[i] - mean_ys)**2
            sum += err
        sum = round(sum, 5)
        print(sum)
        return sum

    def previous_close_price(self, col):
        if col == "Close":
            pcp = float(self.close_values[len(self.close_values) - 1])
        elif col == "High":
            pcp = float(self.high_values[len(self.high_values) - 1])
        elif col == "Low":
            pcp = float(self.low_values[len(self.low_values) - 1])
        else:
            pcp = float(self.open_values[len(self.open_values) - 1])
        print(pcp)
        return pcp

    def recent_price(self, days, col):
        if col == "Close":
            rcp = float(self.close_values[len(self.close_values) - 1 - days])
        elif col == "High":
            rcp = float(self.high_values[len(self.high_values) - 1 - days])
        elif col == "Low":
            rcp = float(self.low_values[len(self.low_values) - 1 - days])
        else:
            rcp = float(self.open_values[len(self.open_values) - 1 - days])
        print(rcp)
        return rcp

def neural_network_test(nn_close, nn_high, nn_low, nn_open, ticker):
    df = aquire_df(ticker)
    true_close_prices = list(df["Close"].tail(30))
    true_high_prices = list(df["High"].tail(30))
    true_low_prices = list(df["Low"].tail(30))
    true_open_prices = list(df["Open"].tail(30))
    normalized_true_close_prices = []
    normalized_true_high_prices = []
    normalized_true_low_prices = []
    normalized_true_open_prices = []
    for i in range(0,len(true_close_prices)):
        normalized_true_close_prices.append(float(float(true_close_prices[i]) / nn_close.z))
        normalized_true_high_prices.append(float(float(true_high_prices[i]) / nn_close.z))
        normalized_true_low_prices.append(float(float(true_low_prices[i]) / nn_close.z))
        normalized_true_open_prices.append(float(float(true_open_prices[i]) / nn_close.z))
    false_close_prices = []
    false_high_prices = []
    false_low_prices = []
    false_open_prices = []
    close_trait = Traits(df, "Close") ############################## Needs to happen inside the for loop
    high_trait = Traits(df, "High")
    low_trait = Traits(df, "Low")
    open_trait = Traits(df, "Open")
    for i in range(0, len(true_close_prices)):
        ### Close
        m30, d30 = close_trait.day_gradient(60, 30)
        m7, d7 = close_trait.day_gradient(37, 30)
        m60_30, d60_30 = close_trait.day_gradient(90, 60)
        m14_7, d14_7 = close_trait.day_gradient(44, 37)
        m21_14, d21_14 = close_trait.day_gradient(51, 44)
        m28_21, d28_21 = close_trait.day_gradient(68, 51)
        sumd30 = close_trait.sos_error(d30)
        sumd7 = close_trait.sos_error(d7)
        sumd60_30 = close_trait.sos_error(d60_30)
        sumd14_7 = close_trait.sos_error(d14_7)
        sumd21_14 = close_trait.sos_error(d21_14)
        sumd28_21 = close_trait.sos_error(d28_21)
        pcp = close_trait.previous_close_price("Close")
        php = close_trait.previous_close_price("High")
        plp = close_trait.previous_close_price("Low")
        pop = close_trait.previous_close_price("Open")
        rcp_2c = close_trait.recent_price(2, "Close")
        rcp_3c = close_trait.recent_price(3, "Close")
        rcp_4c = close_trait.recent_price(4, "Close")
        rcp_7c = close_trait.recent_price(7, "Close")
        rcp_2h = close_trait.recent_price(2, "High")
        rcp_3h = close_trait.recent_price(3, "High")
        rcp_4h = close_trait.recent_price(4, "High")
        rcp_7h = close_trait.recent_price(7, "High")
        rcp_2l = close_trait.recent_price(2, "Low")
        rcp_3l = close_trait.recent_price(3, "Low")
        rcp_4l = close_trait.recent_price(4, "Low")
        rcp_7l = close_trait.recent_price(7, "Low")
        rcp_2o = close_trait.recent_price(2, "Open")
        rcp_3o = close_trait.recent_price(3, "Open")
        rcp_4o = close_trait.recent_price(4, "Open")
        rcp_7o = close_trait.recent_price(7, "Open")
        rcp_14 = close_trait.recent_price(14, "Close")
        rcp_17 = close_trait.recent_price(17, "Close")
        rcp_21 = close_trait.recent_price(21, "Close")
        rcp_41 = close_trait.recent_price(41, "Close")
        rcp_61 = close_trait.recent_price(61, "Close")
        rcp_81 = close_trait.recent_price(81, "Close")
        X_close = [m30, sumd30, m7, sumd7, m60_30, sumd60_30, m14_7, sumd14_7, m21_14, sumd21_14, m28_21,
                         sumd28_21,  pcp, php, plp, pop, rcp_2c, rcp_3c, rcp_4c, rcp_7c, rcp_2h, rcp_3h, rcp_4h, rcp_7h,
                         rcp_2l, rcp_3l, rcp_4l, rcp_7l, rcp_2o, rcp_3o, rcp_4o, rcp_7o,
                         rcp_14, rcp_17, rcp_21, rcp_41, rcp_61, rcp_81]
        nn_close_output = nn_close.think(X_close)

        ### High
        m30, d30 = high_trait.day_gradient(60, 30)
        m7, d7 = high_trait.day_gradient(37, 30)
        m60_30, d60_30 = high_trait.day_gradient(90, 60)
        m14_7, d14_7 = high_trait.day_gradient(44, 37)
        m21_14, d21_14 = high_trait.day_gradient(51, 44)
        m28_21, d28_21 = high_trait.day_gradient(68, 51)
        sumd30 = high_trait.sos_error(d30)
        sumd7 = high_trait.sos_error(d7)
        sumd60_30 = high_trait.sos_error(d60_30)
        sumd14_7 = high_trait.sos_error(d14_7)
        sumd21_14 = high_trait.sos_error(d21_14)
        sumd28_21 = high_trait.sos_error(d28_21)
        pcp = high_trait.previous_close_price("Close")
        php = high_trait.previous_close_price("High")
        plp = high_trait.previous_close_price("Low")
        pop = high_trait.previous_close_price("Open")
        rcp_2c = high_trait.recent_price(2, "Close")
        rcp_3c = high_trait.recent_price(3, "Close")
        rcp_4c = high_trait.recent_price(4, "Close")
        rcp_7c = high_trait.recent_price(7, "Close")
        rcp_2h = high_trait.recent_price(2, "High")
        rcp_3h = high_trait.recent_price(3, "High")
        rcp_4h = high_trait.recent_price(4, "High")
        rcp_7h = high_trait.recent_price(7, "High")
        rcp_2l = high_trait.recent_price(2, "Low")
        rcp_3l = high_trait.recent_price(3, "Low")
        rcp_4l = high_trait.recent_price(4, "Low")
        rcp_7l = high_trait.recent_price(7, "Low")
        rcp_2o = high_trait.recent_price(2, "Open")
        rcp_3o = high_trait.recent_price(3, "Open")
        rcp_4o = high_trait.recent_price(4, "Open")
        rcp_7o = high_trait.recent_price(7, "Open")
        rcp_14 = high_trait.recent_price(14, "Close")
        rcp_17 = high_trait.recent_price(17, "Close")
        rcp_21 = high_trait.recent_price(21, "Close")
        rcp_41 = high_trait.recent_price(41, "Close")
        rcp_61 = high_trait.recent_price(61, "Close")
        rcp_81 = high_trait.recent_price(81, "Close")
        X_high = [m30, sumd30, m7, sumd7, m60_30, sumd60_30, m14_7, sumd14_7, m21_14, sumd21_14, m28_21,
                         sumd28_21,  pcp, php, plp, pop, rcp_2c, rcp_3c, rcp_4c, rcp_7c, rcp_2h, rcp_3h, rcp_4h, rcp_7h,
                         rcp_2l, rcp_3l, rcp_4l, rcp_7l, rcp_2o, rcp_3o, rcp_4o, rcp_7o,
                         rcp_14, rcp_17, rcp_21, rcp_41, rcp_61, rcp_81]
        nn_high_output = nn_high.think(X_high)
        ### Low
        m30, d30 = low_trait.day_gradient(60, 30)
        m7, d7 = low_trait.day_gradient(37, 30)
        m60_30, d60_30 = low_trait.day_gradient(90, 60)
        m14_7, d14_7 = low_trait.day_gradient(44, 37)
        m21_14, d21_14 = low_trait.day_gradient(51, 44)
        m28_21, d28_21 = low_trait.day_gradient(68, 51)
        sumd30 = low_trait.sos_error(d30)
        sumd7 = low_trait.sos_error(d7)
        sumd60_30 = low_trait.sos_error(d60_30)
        sumd14_7 = low_trait.sos_error(d14_7)
        sumd21_14 = low_trait.sos_error(d21_14)
        sumd28_21 = low_trait.sos_error(d28_21)
        pcp = low_trait.previous_close_price("Close")
        php = low_trait.previous_close_price("High")
        plp = low_trait.previous_close_price("Low")
        pop = low_trait.previous_close_price("Open")
        rcp_2c = low_trait.recent_price(2, "Close")
        rcp_3c = low_trait.recent_price(3, "Close")
        rcp_4c = low_trait.recent_price(4, "Close")
        rcp_7c = low_trait.recent_price(7, "Close")
        rcp_2h = low_trait.recent_price(2, "High")
        rcp_3h = low_trait.recent_price(3, "High")
        rcp_4h = low_trait.recent_price(4, "High")
        rcp_7h = low_trait.recent_price(7, "High")
        rcp_2l = low_trait.recent_price(2, "Low")
        rcp_3l = low_trait.recent_price(3, "Low")
        rcp_4l = low_trait.recent_price(4, "Low")
        rcp_7l = low_trait.recent_price(7, "Low")
        rcp_2o = low_trait.recent_price(2, "Open")
        rcp_3o = low_trait.recent_price(3, "Open")
        rcp_4o = low_trait.recent_price(4, "Open")
        rcp_7o = low_trait.recent_price(7, "Open")
        rcp_14 = low_trait.recent_price(14, "Close")
        rcp_17 = low_trait.recent_price(17, "Close")
        rcp_21 = low_trait.recent_price(21, "Close")
        rcp_41 = low_trait.recent_price(41, "Close")
        rcp_61 = low_trait.recent_price(61, "Close")
        rcp_81 = low_trait.recent_price(81, "Close")
        X_low = [m30, sumd30, m7, sumd7, m60_30, sumd60_30, m14_7, sumd14_7, m21_14, sumd21_14, m28_21,
                         sumd28_21,  pcp, php, plp, pop, rcp_2c, rcp_3c, rcp_4c, rcp_7c, rcp_2h, rcp_3h, rcp_4h, rcp_7h,
                         rcp_2l, rcp_3l, rcp_4l, rcp_7l, rcp_2o, rcp_3o, rcp_4o, rcp_7o,
                         rcp_14, rcp_17, rcp_21, rcp_41, rcp_61, rcp_81]
        nn_low_output = nn_low.think(X_low)
        ### Open
        m30, d30 = open_trait.day_gradient(60, 30)
        m7, d7 = open_trait.day_gradient(37, 30)
        m60_30, d60_30 = open_trait.day_gradient(90, 60)
        m14_7, d14_7 = open_trait.day_gradient(44, 37)
        m21_14, d21_14 = open_trait.day_gradient(51, 44)
        m28_21, d28_21 = open_trait.day_gradient(68, 51)
        sumd30 = open_trait.sos_error(d30)
        sumd7 = open_trait.sos_error(d7)
        sumd60_30 = open_trait.sos_error(d60_30)
        sumd14_7 = open_trait.sos_error(d14_7)
        sumd21_14 = open_trait.sos_error(d21_14)
        sumd28_21 = open_trait.sos_error(d28_21)
        pcp = open_trait.previous_close_price("Close")
        php = open_trait.previous_close_price("High")
        plp = open_trait.previous_close_price("Low")
        pop = open_trait.previous_close_price("Open")
        rcp_2c = open_trait.recent_price(2, "Close")
        rcp_3c = open_trait.recent_price(3, "Close")
        rcp_4c = open_trait.recent_price(4, "Close")
        rcp_7c = open_trait.recent_price(7, "Close")
        rcp_2h = open_trait.recent_price(2, "High")
        rcp_3h = open_trait.recent_price(3, "High")
        rcp_4h = open_trait.recent_price(4, "High")
        rcp_7h = open_trait.recent_price(7, "High")
        rcp_2l = open_trait.recent_price(2, "Low")
        rcp_3l = open_trait.recent_price(3, "Low")
        rcp_4l = open_trait.recent_price(4, "Low")
        rcp_7l = open_trait.recent_price(7, "Low")
        rcp_2o = open_trait.recent_price(2, "Open")
        rcp_3o = open_trait.recent_price(3, "Open")
        rcp_4o = open_trait.recent_price(4, "Open")
        rcp_7o = open_trait.recent_price(7, "Open")
        rcp_14 = open_trait.recent_price(14, "Close")
        rcp_17 = open_trait.recent_price(17, "Close")
        rcp_21 = open_trait.recent_price(21, "Close")
        rcp_41 = open_trait.recent_price(41, "Close")
        rcp_61 = open_trait.recent_price(61, "Close")
        rcp_81 = open_trait.recent_price(81, "Close")
        X_open = [m30, sumd30, m7, sumd7, m60_30, sumd60_30, m14_7, sumd14_7, m21_14, sumd21_14, m28_21,
                         sumd28_21,  pcp, php, plp, pop, rcp_2c, rcp_3c, rcp_4c, rcp_7c, rcp_2h, rcp_3h, rcp_4h, rcp_7h,
                         rcp_2l, rcp_3l, rcp_4l, rcp_7l, rcp_2o, rcp_3o, rcp_4o, rcp_7o,
                         rcp_14, rcp_17, rcp_21, rcp_41, rcp_61, rcp_81]
        nn_open_output = nn_open.think(X_open)

        false_close_prices.append(nn_close_output[0])
        false_high_prices.append(nn_high_output[0])
        false_low_prices.append(nn_low_output[0])
        false_open_prices.append(nn_open_output[0])
        ## for close_traits
        # remove last day
        close_trait.close_values = close_trait.close_values[1:]
        close_trait.high_values = close_trait.high_values[1:]
        close_trait.low_values = close_trait.low_values[1:]
        close_trait.open_values = close_trait.open_values[1:]
        # add newday
        close_trait.close_values = np.append(close_trait.close_values, nn_close.denormalize_data(nn_close_output))
        close_trait.high_values = np.append(close_trait.high_values, nn_high.denormalize_data(nn_high_output))
        close_trait.low_values = np.append(close_trait.low_values, nn_low.denormalize_data(nn_low_output))
        close_trait.open_values = np.append(close_trait.open_values, nn_open.denormalize_data(nn_open_output))

        ## for high_traits
        # remove last day
        high_trait.close_values = high_trait.close_values[1:]
        high_trait.high_values = high_trait.high_values[1:]
        high_trait.low_values = high_trait.low_values[1:]
        high_trait.open_values = high_trait.open_values[1:]
        # add newday
        high_trait.close_values = np.append(high_trait.close_values, nn_close.denormalize_data(nn_close_output))
        high_trait.high_values = np.append(high_trait.high_values, nn_high.denormalize_data(nn_high_output))
        high_trait.low_values = np.append(high_trait.low_values, nn_low.denormalize_data(nn_low_output))
        high_trait.open_values = np.append(high_trait.open_values, nn_open.denormalize_data(nn_open_output))

        ## for low_traits
        # remove last day
        low_trait.close_values = low_trait.close_values[1:]
        low_trait.high_values = low_trait.high_values[1:]
        low_trait.low_values = low_trait.low_values[1:]
        low_trait.open_values = low_trait.open_values[1:]
        # add newday
        low_trait.close_values = np.append(low_trait.close_values, nn_close.denormalize_data(nn_close_output))
        low_trait.high_values = np.append(low_trait.high_values, nn_high.denormalize_data(nn_high_output))
        low_trait.low_values = np.append(low_trait.low_values, nn_low.denormalize_data(nn_low_output))
        low_trait.open_values = np.append(low_trait.open_values, nn_open.denormalize_data(nn_open_output))

        ## for low_traits
        # remove last day
        open_trait.close_values = open_trait.close_values[1:]
        open_trait.high_values = open_trait.high_values[1:]
        open_trait.low_values = open_trait.low_values[1:]
        open_trait.open_values = open_trait.open_values[1:]
        # add newday
        open_trait.close_values = np.append(open_trait.close_values, nn_close.denormalize_data(nn_close_output))
        open_trait.high_values = np.append(open_trait.high_values, nn_high.denormalize_data(nn_high_output))
        open_trait.low_values = np.append(open_trait.low_values, nn_low.denormalize_data(nn_low_output))
        open_trait.open_values = np.append(open_trait.open_values, nn_open.denormalize_data(nn_open_output))


        # print(false_prices)
    # print(normalized_true_prices)
    # print("actual_prices")
    # ai_prices = [nn.denormalize_data(i) for i in false_prices]
    # print(ai_prices)
    # print(true_prices)
    # print("pcp is " + str(test_trait.previous_close_price()))
    # print(test_trait.close_values)
    # print(len(test_trait.close_values))
    # plt.plot(false_prices, label="AI_prices")
    # plt.plot(normalized_true_prices, label="True_price")
    # plt.legend()
    # plt.show()
    return list(close_trait.close_values[-30:]), list(high_trait.high_values[-30:]), list(low_trait.low_values[-30:]), list(open_trait.open_values[-30:])


def creating_classifying_set(df_col="Close"):
    """
    The function is training the nn based on the col needing data
    :param df_col: String, name of column
    :return: Class, a neural network.
    """
    with open('tickers.txt') as f:
        all_x = []
        first_y = []
        all_y = []
        count = 0
        for line in f:
            try:
                line = line.replace('\r', '').replace('\n', '')
                line = line[:3].lower()
                df = aquire_df(line)
                first_trait = Traits(df, df_col)
                if float(df[df_col].tail(1)) > 0.1:
                    m30, d30 = first_trait.day_gradient(30, 0)
                    m7, d7 = first_trait.day_gradient(7, 0)
                    m60_30, d60_30 = first_trait.day_gradient(60, 30)
                    m14_7, d14_7 = first_trait.day_gradient(14, 7)
                    m21_14, d21_14 = first_trait.day_gradient(21, 14)
                    m28_21, d28_21 = first_trait.day_gradient(28, 21)
                    sumd30 = first_trait.sos_error(d30)
                    sumd7 = first_trait.sos_error(d7)
                    sumd60_30 = first_trait.sos_error(d60_30)
                    sumd14_7 = first_trait.sos_error(d14_7)
                    sumd21_14 = first_trait.sos_error(d21_14)
                    sumd28_21 = first_trait.sos_error(d28_21)
                    pcp = first_trait.previous_close_price("Close")
                    php = first_trait.previous_close_price("High")
                    plp = first_trait.previous_close_price("Low")
                    pop = first_trait.previous_close_price("Open")
                    rcp_2c = first_trait.recent_price(2, "Close")
                    rcp_3c = first_trait.recent_price(3, "Close")
                    rcp_4c = first_trait.recent_price(4, "Close")
                    rcp_7c = first_trait.recent_price(7, "Close")
                    rcp_2h = first_trait.recent_price(2, "High")
                    rcp_3h = first_trait.recent_price(3, "High")
                    rcp_4h = first_trait.recent_price(4, "High")
                    rcp_7h = first_trait.recent_price(7, "High")
                    rcp_2l = first_trait.recent_price(2, "Low")
                    rcp_3l = first_trait.recent_price(3, "Low")
                    rcp_4l = first_trait.recent_price(4, "Low")
                    rcp_7l = first_trait.recent_price(7, "Low")
                    rcp_2o = first_trait.recent_price(2, "Open")
                    rcp_3o = first_trait.recent_price(3, "Open")
                    rcp_4o = first_trait.recent_price(4, "Open")
                    rcp_7o = first_trait.recent_price(7, "Open")
                    rcp_14 = first_trait.recent_price(14, "Close")
                    rcp_17 = first_trait.recent_price(17, "Close")
                    rcp_21 = first_trait.recent_price(21, "Close")
                    rcp_41 = first_trait.recent_price(41, "Close")
                    rcp_61 = first_trait.recent_price(61, "Close")
                    rcp_81 = first_trait.recent_price(81, "Close")
                    X = [m30, sumd30, m7, sumd7, m60_30, sumd60_30, m14_7, sumd14_7, m21_14, sumd21_14, m28_21,
                         sumd28_21,  pcp, php, plp, pop, rcp_2c, rcp_3c, rcp_4c, rcp_7c, rcp_2h, rcp_3h, rcp_4h, rcp_7h,
                         rcp_2l, rcp_3l, rcp_4l, rcp_7l, rcp_2o, rcp_3o, rcp_4o, rcp_7o,
                         rcp_14, rcp_17, rcp_21, rcp_41, rcp_61, rcp_81]
                    #tmp = list(df["Close"].tail(1), df["High"].tail(1), df["Low"].tail(1), df["Open"].tail(1))
                    tmp = list(df[df_col].tail(1))
                    Y = float(tmp[0])
                    all_x.append(X)
                    first_y.append(Y)
                    #print("Done:" + line)
                else:
                    print("-----------------------------------------------Too small-------------------------------------")
            except:
                print("ERROR! ticker doesn't have full data")
            count += 1
            if count > 500:
                break
        all_y.append(first_y)
        print(all_y)
    nn = NeuralNetwork(np.array(all_x), np.array(all_y).T)
    total_sum = []
    #print(np.array(all_x).shape)
    #print(np.array(all_y).T.shape)
    for i in range(2500):
        nn.feedforward()
        #print(nn.output)
        nn.backprop()
        sum = 0
        for i in range(0, len(all_y)):
            err = (nn.output[len(all_y)-1, i] - nn.y[i]) ** 2
            sum += err
        total_sum.append(sum)
    # plt.plot(total_sum)
    # plt.show()
    # print(nn.output)
    # print(nn.y)
    # neural_network_test(nn, "ddr")
    return nn

def candlestick_AI():
    high_nn = creating_classifying_set(df_col="High")
    low_nn = creating_classifying_set(df_col="Low")
    close_nn = creating_classifying_set(df_col="Close")
    open_nn = creating_classifying_set(df_col="Open")

    close_list, high_list, low_list, open_list = neural_network_test(close_nn, high_nn, low_nn, open_nn, "rbl")

    plt.plot(high_list, label="high")
    plt.plot(low_list, label="low")
    plt.plot(open_list, label="open")
    plt.plot(close_list, label="close")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    candlestick_AI()
    # creating_classifying_set()

    # X = np.array([[0, 0, 1],
    #               [0, 1, 1],
    #               [1, 0, 1],
    #               [1, 1, 1]])
    # y = np.array([[0], [1], [1], [0]])
    # nn = NeuralNetwork(X, y)
    #
    # for i in range(1500):
    #     nn.feedforward()
    #     nn.backprop()
    #
    # print(nn.output)
    # print(X.shape)
    # print(y.shape)

    # k = aquire_df("ddr")
    # m, y = thirty_day_gradient(k)
    # thirty_day_sos_error(y)
    # k = aquire_df("esh")
    # m, y = thirty_day_gradient(k)
    # thirty_day_sos_error(y)
    # k = aquire_df("bhp")
    # m, y = thirty_day_gradient(k)
    # thirty_day_sos_error(y)
#
# def sigmoid(x):
#     return 1.0/(1+ np.exp(-x))
#
# def sigmoid_derivative(x):
#     return x * (1.0 - x)
#
#
# class NeuralNetwork:
#     '''
#     This is the original Neural network
#     '''
#     def __init__(self, x, y):
#         self.input      = x
#         self.weights1   = np.random.rand(self.input.shape[1],4)
#         self.weights2   = np.random.rand(4,1)
#         self.y          = y
#         self.output     = np.zeros(self.y.shape)
#
#     def feedforward(self):
#         self.layer1 = sigmoid(np.dot(self.input, self.weights1))
#         self.output = sigmoid(np.dot(self.layer1, self.weights2))
#
#     def backprop(self):
#         # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
#         d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
#         d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))
#
#         # update the weights with the derivative (slope) of the loss function
#         self.weights1 += d_weights1
#         self.weights2 += d_weights2
#
#
# if __name__ == "__main__":
#     X = np.array([[0,0,1],
#                   [0,1,1],
#                   [1,0,1],
#                   [1,1,1]])
#     y = np.array([[0],[1],[1],[0]])
#     nn = NeuralNetwork(X,y)
#
#     for i in range(1500):
#         nn.feedforward()
#         nn.backprop()
#
#     print(nn.output)
