import pandas as pd
import numpy as np
import os
os.makedirs(data_dir, exist_ok=True)
os.makedirs(npy_newsdata_dir, exist_ok=True)

def csv_import(data_dir, csv_filename):
    data_pd = pd.read_csv(data_dir + csv_filename)
    data_pd['Predict'] = np.zeros(len(data_pd['Date']))
    return data_pd

def Gaussian_noise(data_pd, lamda):
    row = data_pd.shape[0]
    std = 0
    std = data_pd.std()
    data_pd.loc[:] = data_pd.loc[:] + np.random.normal(loc = 0.0, scale = lamda * std, size = row)

    return data_pd

def data_normalization(data_pd, stock_price, stock_trade, stock_news, lamda):
    price = pd.concat(data_pd.loc[:, k] for k in stock_price)
    data_pd[stock_price] = data_pd[stock_price].applymap(lambda x: (x - price.min()) / (price.max() - price.min()))
    trade = pd.concat(data_pd.loc[:, k] for k in stock_trade)
    data_pd[stock_trade] = data_pd[stock_trade].applymap(lambda x: (x - trade.min()) / (trade.max() - trade.min()))

    data_pd.loc[:, 'cnbc_data'] = Gaussian_noise(data_pd.loc[:, 'cnbc_data'], lamda)
    data_pd.loc[:, 'fortune_data'] = Gaussian_noise(data_pd.loc[:, 'fortune_data'], lamda)
    data_pd.loc[:, 'reuters_data'] = Gaussian_noise(data_pd.loc[:, 'reuters_data'], lamda)
    data_pd.loc[:, 'wsj_data'] = Gaussian_noise(data_pd.loc[:, 'wsj_data'], lamda)

    news = pd.concat(data_pd.loc[:, k] for k in stock_news)
    data_pd[stock_news] = data_pd[stock_news].applymap(lambda x: (x - news.min()) / (news.max() - news.min()))
    data_pd['Predict'] = data_pd['Close']
    print(price.max())
    print(price.min())
    print(news.max())
    print(news.min())
    return data_pd

def data_numbers(data_pd, train_date):
    i = 0
    j = 0
    for k in range(len(data_pd['Date'])):
        if data_pd.loc[k, 'Date'] <= train_date:
            i += 1
        if data_pd.loc[k, 'Date'] > train_date:
            j += 1
    return i, j

def LSTM_data_to_tensor(data_pd, train_date, days, factor_num):
    train_days, test_days = data_numbers(data_pd, train_date)

    train_in = np.zeros((train_days - days, days, factor_num))
    train_out = np.zeros((train_days - days, 1))
    test_in = np.zeros((test_days, days, factor_num))
    test_out = np.zeros((test_days, 1))

    Train = 0
    Test = 0
    for i in range(len(data_pd['Date']) - days):
        if data_pd.loc[i + days, 'Date'] <= train_date:
            train_in[Train] = data_pd.loc[i:i + days - 1, 'Open':'wsj_data']
            train_out[Train] = data_pd.loc[i + days, 'Predict']
            Train += 1
        elif data_pd.loc[i + days, 'Date'] > train_date:
            test_in[Test] = data_pd.loc[i:i + days - 1, 'Open':'wsj_data']
            test_out[Test] = data_pd.loc[i + days, 'Predict']
            Test += 1

    print(train_in.shape)
    print(test_in.shape)
    return train_in, train_out, test_in, test_out

if __name__ == '__main__':
    days = 5
    train_date = 20180507
    factor_num = 9
    lamda = 0.005
    stock_price = ['Open', 'High', 'Low', 'Close']
    stock_trade = ['Volume']
    stock_news = ['cnbc_data', 'fortune_data', 'reuters_data', 'wsj_data']

    data_pd = csv_import(data_dir, 'SP500_news.csv')
    data_pd = data_normalization(data_pd, stock_price, stock_trade, stock_news, lamda)
    train_days, test_days = data_numbers(data_pd, train_date)

    data_pd.to_csv(data_dir + 'normalization_SP500_news.csv')

    train_in, train_out, test_in, test_out = LSTM_data_to_tensor(data_pd, train_date, days, factor_num)

    np.save(npy_newsdata_dir + 'train_in_SP500_news', train_in)
    np.save(npy_newsdata_dir + 'train_out_SP500_news', train_out)
    np.save(npy_newsdata_dir + 'test_in_SP500_news', test_in)
    np.save(npy_newsdata_dir + 'test_out_SP500_news', test_out)
