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
        self.input      = x
        self.train_mean = np.mean(x)
        self.train_std  = np.std(x)
        self.stdize     = (x - self.train_mean)/self.train_std
        self.z          = np.array([6.4])
        self.weights1   = np.random.rand(self.input.shape[1], 29)
        self.weights2   = np.random.rand(29, 1)
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

    def static_think_2500(self, inputs):
        nn_2500_weights1 = np.load('nn_2500_weights1.npy')
        nn_2500_weights2 = np.load('nn_2500_weights2.npy')
        nn_2500_train_mean = np.load('nn_2500_train_mean.npy')
        nn_2500_train_std = np.load('nn_2500_train_std.npy')
        std_value = (inputs - nn_2500_train_mean) / nn_2500_train_std
        layer1 = sigmoid(np.dot(std_value, nn_2500_weights1))
        output = sigmoid(np.dot(layer1, nn_2500_weights2))
        return output

    def static_think_2000(self, inputs):
        nn_2000_weights1 = np.load('nn_2000_weights1.npy')
        nn_2000_weights2 = np.load('nn_2000_weights2.npy')
        nn_2000_train_mean = np.load('nn_2000_train_mean.npy')
        nn_2000_train_std = np.load('nn_2000_train_std.npy')
        std_value = (inputs - nn_2000_train_mean) / nn_2000_train_std
        layer1 = sigmoid(np.dot(std_value, nn_2000_weights1))
        output = sigmoid(np.dot(layer1, nn_2000_weights2))
        return output

    def static_think_1500(self, inputs):
        nn_1500_weights1 = np.load('nn_1500_weights1.npy')
        nn_1500_weights2 = np.load('nn_1500_weights2.npy')
        nn_1500_train_mean = np.load('nn_1500_train_mean.npy')
        nn_1500_train_std = np.load('nn_1500_train_std.npy')
        std_value = (inputs - nn_1500_train_mean) / nn_1500_train_std
        layer1 = sigmoid(np.dot(std_value, nn_1500_weights1))
        output = sigmoid(np.dot(layer1, nn_1500_weights2))
        return output

    def static_think_1000(self, inputs):
        nn_1000_weights1 = np.load('nn_1000_weights1.npy')
        nn_1000_weights2 = np.load('nn_1000_weights2.npy')
        nn_1000_train_mean = np.load('nn_1000_train_mean.npy')
        nn_1000_train_std = np.load('nn_1000_train_std.npy')
        std_value = (inputs - nn_1000_train_mean) / nn_1000_train_std
        layer1 = sigmoid(np.dot(std_value, nn_1000_weights1))
        output = sigmoid(np.dot(layer1, nn_1000_weights2))
        return output

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


    def day_gradient(self, start, remove_days=0):
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

    def previous_close_price(self):
        pcp = float(self.close_values[len(self.close_values) - 1])
        print(pcp)
        return pcp

    def recent_price(self, days):
        rcp = float(self.close_values[len(self.close_values) - 1 - days])
        print(rcp)
        return rcp

def neural_network_test(nn_2500, nn_2000, nn_1500, nn_1000, ticker):
    df = aquire_df(ticker)
    true_close_prices = list(df["Close"].tail(30))
    normalized_true_close_prices = []
    for i in range(0,len(true_close_prices)):
        normalized_true_close_prices.append(float(float(true_close_prices[i]) / nn_2500.z))
    false_nn_2500_prices = []
    false_nn_2000_prices = []
    false_nn_1500_prices = []
    false_nn_1000_prices = []
    close_trait = Traits(df, "Close") ############################## Needs to happen inside the for loop
    print("#############")
    print(close_trait.previous_close_price())
    print(close_trait.recent_price(2))
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
        pcp = close_trait.previous_close_price()
        rcp_2c = close_trait.recent_price(2)
        rcp_3c = close_trait.recent_price(3)
        rcp_4c = close_trait.recent_price(4)
        rcp_7c = close_trait.recent_price(7)
        rcp_14 = close_trait.recent_price(14)
        rcp_17 = close_trait.recent_price(17)
        rcp_21 = close_trait.recent_price(21)
        rcp_41 = close_trait.recent_price(41)
        rcp_61 = close_trait.recent_price(61)
        rcp_81 = close_trait.recent_price(81)
        X_close = [m30, sumd30, m7, sumd7, m60_30, sumd60_30, m14_7, sumd14_7, m21_14, sumd21_14, m28_21,
                         sumd28_21,  pcp, rcp_2c, rcp_3c, rcp_4c, rcp_7c,
                         rcp_14, rcp_17, rcp_21, rcp_41, rcp_61, rcp_81]
        nn_2500_output = nn_2500.static_think_2500(X_close)
        nn_2000_output = nn_2000.static_think_2000(X_close)
        nn_1500_output = nn_1500.static_think_1500(X_close)
        nn_1000_output = nn_1000.static_think_1000(X_close)
        #nn_2500_output = nn_2500(X_close)
        #nn_2000_output = nn_2000.think(X_close)
        #nn_1500_output = nn_1500.think(X_close)
        #nn_1000_output = nn_1000.think(X_close)
        false_nn_2500_prices.append(nn_2500_output[0])
        false_nn_2000_prices.append(nn_2000_output[0])
        false_nn_1500_prices.append(nn_1500_output[0])
        false_nn_1000_prices.append(nn_1000_output[0])
        ## for close_traits
        # remove last day
        close_trait.close_values = close_trait.close_values[1:]
        # add newday
        close_trait.close_values = np.append(close_trait.close_values, nn_2500.denormalize_data(nn_2500_output))

    ai_prices_nn_2500 = [nn_2500.denormalize_data(i) for i in false_nn_2500_prices]
    ai_prices_nn_2000 = [nn_2500.denormalize_data(i) for i in false_nn_2000_prices]
    ai_prices_nn_1500 = [nn_2500.denormalize_data(i) for i in false_nn_1500_prices]
    ai_prices_nn_1000 = [nn_2500.denormalize_data(i) for i in false_nn_1000_prices] # perhaps has to be denormalize by own neural network
    print(ai_prices_nn_2500)
    print(true_close_prices)
    print("pcp is " + str(close_trait.previous_close_price()))
    print(close_trait.close_values)
    print(len(close_trait.close_values))
    plt.plot(ai_prices_nn_2500, label="nn_2500")
    plt.plot(ai_prices_nn_2000, label="nn_2000")
    plt.plot(ai_prices_nn_1500, label="nn_1500")
    plt.plot(ai_prices_nn_1000, label="nn_1000")
    plt.plot(true_close_prices, label="True_price")
    plt.legend()
    plt.show()
    return list(close_trait.close_values[-30:])


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
                    pcp = first_trait.previous_close_price()
                    rcp_2c = first_trait.recent_price(2)
                    rcp_3c = first_trait.recent_price(3)
                    rcp_4c = first_trait.recent_price(4)
                    rcp_7c = first_trait.recent_price(7)
                    rcp_14 = first_trait.recent_price(14)
                    rcp_17 = first_trait.recent_price(17)
                    rcp_21 = first_trait.recent_price(21)
                    rcp_41 = first_trait.recent_price(41)
                    rcp_61 = first_trait.recent_price(61)
                    rcp_81 = first_trait.recent_price(81)
                    X = [m30, sumd30, m7, sumd7, m60_30, sumd60_30, m14_7, sumd14_7, m21_14, sumd21_14, m28_21,
                         sumd28_21,  pcp, rcp_2c, rcp_3c, rcp_4c, rcp_7c,
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
    nn_2500 = NeuralNetwork(np.array(all_x), np.array(all_y).T)
    nn_2000 = NeuralNetwork(np.array(all_x), np.array(all_y).T)
    nn_1500 = NeuralNetwork(np.array(all_x), np.array(all_y).T)
    nn_1000 = NeuralNetwork(np.array(all_x), np.array(all_y).T)
    total_sum = []
    #print(np.array(all_x).shape)
    #print(np.array(all_y).T.shape)
    for i in range(2500):
        nn_2500.feedforward()
        #print(nn.output)
        nn_2500.backprop()
        if i < 2000:
            nn_2000.feedforward()
            nn_2000.backprop()
        if i < 1500:
            nn_1500.feedforward()
            nn_1500.backprop()
        if i < 1000:
            nn_1000.feedforward()
            nn_1000.backprop()
        sum = 0
        for i in range(0, len(all_y)):
            err = (nn_2500.output[len(all_y)-1, i] - nn_2500.y[i]) ** 2
            sum += err
        total_sum.append(sum)
    # plt.plot(total_sum)
    # plt.show()
    # print(nn.output)
    # print(nn.y)
    # neural_network_test(nn, "ddr")
    return nn_2500, nn_2000, nn_1500, nn_1000

def alpha_candlestick_AI():
    nn_2500, nn_2000, nn_1500, nn_1000 = creating_classifying_set(df_col="Close")
    close_list = neural_network_test(nn_2500, nn_2000, nn_1500, nn_1000, "ddr")
    print(nn_2500.train_mean.shape)
    print(nn_2500.train_std.shape)
    np.save('nn_2500_weights1.npy', nn_2500.weights1)
    np.save('nn_2500_weights2.npy', nn_2500.weights2)
    np.save('nn_2500_train_mean.npy', nn_2500.train_mean)
    np.save('nn_2500_train_std.npy', nn_2500.train_std)
    np.save('nn_2000_weights1.npy', nn_2000.weights1)
    np.save('nn_2000_weights2.npy', nn_2000.weights2)
    np.save('nn_2000_train_mean.npy', nn_2000.train_mean)
    np.save('nn_2000_train_std.npy', nn_2000.train_std)
    np.save('nn_1500_weights1.npy', nn_1500.weights1)
    np.save('nn_1500_weights2.npy', nn_1500.weights2)
    np.save('nn_1500_train_mean.npy', nn_1500.train_mean)
    np.save('nn_1500_train_std.npy', nn_1500.train_std)
    np.save('nn_1000_weights1.npy', nn_1000.weights1)
    np.save('nn_1000_weights2.npy', nn_1000.weights2)
    np.save('nn_1000_train_mean.npy', nn_1000.train_mean)
    np.save('nn_1000_train_std.npy', nn_1000.train_std)
    print(nn_2500.weights1)
    print(nn_2500.weights2)
    plt.plot(close_list, label="close")
    plt.legend()
    plt.show()

def beta_candlestick_AI(ticker):
    nn_dummy = NeuralNetwork(np.ones([29, 1], dtype=int), np.ones([29, 1], dtype=int))
    df = aquire_df(ticker)
    true_close_prices = list(df["Close"].tail(30))
    normalized_true_close_prices = []
    for i in range(0, len(true_close_prices)):
        normalized_true_close_prices.append(float(float(true_close_prices[i]) / nn_dummy.z))
    false_nn_2500_prices = []
    false_nn_2000_prices = []
    false_nn_1500_prices = []
    false_nn_1000_prices = []
    close_trait = Traits(df, "Close")  ############################## Needs to happen inside the for loop

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
        pcp = close_trait.previous_close_price()
        rcp_2c = close_trait.recent_price(2)
        rcp_3c = close_trait.recent_price(3)
        rcp_4c = close_trait.recent_price(4)
        rcp_7c = close_trait.recent_price(7)
        rcp_14 = close_trait.recent_price(14)
        rcp_17 = close_trait.recent_price(17)
        rcp_21 = close_trait.recent_price(21)
        rcp_41 = close_trait.recent_price(41)
        rcp_61 = close_trait.recent_price(61)
        rcp_81 = close_trait.recent_price(81)
        X_close = [m30, sumd30, m7, sumd7, m60_30, sumd60_30, m14_7, sumd14_7, m21_14, sumd21_14, m28_21,
                   sumd28_21, pcp, rcp_2c, rcp_3c, rcp_4c, rcp_7c,
                   rcp_14, rcp_17, rcp_21, rcp_41, rcp_61, rcp_81]
        nn_2500_output = nn_dummy.static_think_2500(X_close)
        nn_2000_output = nn_dummy.static_think_2000(X_close)
        nn_1500_output = nn_dummy.static_think_1500(X_close)
        nn_1000_output = nn_dummy.static_think_1000(X_close)
        false_nn_2500_prices.append(nn_2500_output[0])
        false_nn_2000_prices.append(nn_2000_output[0])
        false_nn_1500_prices.append(nn_1500_output[0])
        false_nn_1000_prices.append(nn_1000_output[0])
        ## for close_traits
        # remove last day
        close_trait.close_values = close_trait.close_values[1:]
        # add newday
        close_trait.close_values = np.append(close_trait.close_values, nn_dummy.denormalize_data(nn_2500_output))

    ai_prices_nn_2500 = [nn_dummy.denormalize_data(i) for i in false_nn_2500_prices]
    ai_prices_nn_2000 = [nn_dummy.denormalize_data(i) for i in false_nn_2000_prices]
    ai_prices_nn_1500 = [nn_dummy.denormalize_data(i) for i in false_nn_1500_prices]
    ai_prices_nn_1000 = [nn_dummy.denormalize_data(i) for i in false_nn_1000_prices]  # perhaps has to be denormalize by own neural network
    low_list = []
    high_list = []
    open_list = []
    close_list = []
    low_list.append( min([ai_prices_nn_2500[0], ai_prices_nn_2000[0], ai_prices_nn_1500[0], ai_prices_nn_1000[0]]))
    high_list.append(max([ai_prices_nn_2500[0], ai_prices_nn_2000[0], ai_prices_nn_1500[0], ai_prices_nn_1000[0]]))
    close_list.append(ai_prices_nn_2000[0])
    open_list.append(ai_prices_nn_1500[0])
    #Arrange estimates into correct categories
    for i in range(1, len(ai_prices_nn_2500)):
        low = min([ai_prices_nn_2500[i], ai_prices_nn_2000[i], ai_prices_nn_1500[i], ai_prices_nn_1000[i]])
        high = max([ai_prices_nn_2500[i], ai_prices_nn_2000[i], ai_prices_nn_1500[i], ai_prices_nn_1000[i]])
        p2h = second_largest([ai_prices_nn_2500[i], ai_prices_nn_2000[i], ai_prices_nn_1500[i], ai_prices_nn_1000[i]]) #possible 2nd highest
        p2l = second_smallest([ai_prices_nn_2500[i], ai_prices_nn_2000[i], ai_prices_nn_1500[i], ai_prices_nn_1000[i]]) #possible 2nd lowest
        low_list.append(low)
        high_list.append(high)
        if (p2h+p2l)/2 >close_list[i-1]:
            close_list.append(p2h)
            open_list.append(p2l)
        else:
            close_list.append(p2l)
            open_list.append(p2h)
    return close_list, high_list, low_list, open_list
    #
    # print(ai_prices_nn_2500)
    # print(true_close_prices)
    # print("pcp is " + str(close_trait.previous_close_price()))
    # print(close_trait.close_values)
    # print(len(close_trait.close_values))
    # plt.plot(ai_prices_nn_2500, label="nn_2500")
    # plt.plot(ai_prices_nn_2000, label="nn_2000")
    # plt.plot(ai_prices_nn_1500, label="nn_1500")
    # plt.plot(ai_prices_nn_1000, label="nn_1000")
    # plt.plot(true_close_prices, label="True_price")
    # plt.legend()
    # plt.show()

def second_largest(numbers):
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1
            else:
                m2 = x
    return m2 if count >= 2 else None

def second_smallest(numbers):
    count = 0
    ms = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x < m2:
            if x <= ms:
                ms, m2 = x, ms
            else:
                m2 = x
    return m2 if count >= 2 else None


if __name__ == "__main__":
    beta_candlestick_AI("ddr")
