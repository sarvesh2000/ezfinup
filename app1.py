from flask import Flask, render_template,request
import os, csv
import numpy as np
import pandas_datareader as pdr
import pandas as pd
import matplotlib.pyplot as plt
#import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from newsapi import NewsApiClient
import talib
import yfinance as yf
from patterns import candlestick_patterns

# Init
newsapi = NewsApiClient(api_key='0144a0f2461949b7b8896463a90399f3')


key="483bd89d87mshce534f1bd4ba6b0p182fcdjsnf08d8fc19c04"


def get_company_name(symbol):
    if symbol== 'AMZN':
        return 'Amazon'
    elif symbol == 'TSLA':
        return 'Tesla'
    elif symbol=='GOOG':
        return 'Aplhabet'
    elif symbol=='AAPL':
        return 'Apple'
    elif symbol=='IDEA.NS':
    	return 'IDEA.NS'
    else:
        'None'

def get_data(symbol,start,end):
    
    #load the data
    if symbol.upper() == 'AMZN':
        df = pd.read_csv("AMZN.csv")
    elif symbol.upper() == 'TSLA':
        df = pd.read_csv("TSLA.csv")
    elif symbol.upper() == 'GOOG':
        df = pd.read_csv("GOOG.csv")  
    elif symbol.upper() == 'AAPL':
        df = pd.read_csv("AAPL.csv") 
    elif symbol.upper() == 'IDEA.NS':
        df = pd.read_csv("IDEANS.csv")

   #get the date range
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    #Set the start and end index rows both to 0
    start_row=0
    end_row=0

    #start the date from the top of the dataset and go down to see if the users start date<=date in the dataset
    for i in range(0,len(df)):
        if start <= pd.to_datetime(df['Date'][i]):
            start_row = i
            break
    #start from the bottom of the dataset and go up to see if the users end date is greater or equal to the date in the dataset\
    for j in range(0,len(df)):
        if end >= pd.to_datetime(df['Date'][len(df)-1-j]):
            end_row = len(df)-1-j
            break
    #set the index to the date
    df=df.set_index(pd.DatetimeIndex(df['Date'].values))
    
    return df.iloc[start_row:end_row+1, : ]

def predictIdea():
    df = pdr.get_data_yahoo('IDEA.NS')
    df1 = df. reset_index()['Close']

    import numpy as np

    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

    training_size=int(len(df1)*0.65)
    test_size=len(df1)-training_size
    train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

    import numpy
    # convert an array of values into a dataset matrix
    def create_dataset(dataset, time_step=1):
    	dataX, dataY = [], []
    	for i in range(len(dataset)-time_step-1):
    		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
    		dataX.append(a)
    		dataY.append(dataset[i + time_step, 0])
    	return numpy.array(dataX), numpy.array(dataY)

    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    ### Create the Stacked LSTM model
    import tensorflow as tf
    import numpy as np
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import LSTM

    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))


    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')

    model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=10,batch_size=64,verbose=1)

    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)

    ##Transformback to original form --- rescaling
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)

    ### Plotting 
    # shift train predictions for plotting
    look_back=100
    trainPredictPlot = numpy.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(df1)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict

    x_input = test_data[331:].reshape(1,-1)

    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    # demonstrate prediction for next 10 days
    from numpy import array

    lst_output=[]
    n_steps=100
    i=0
    while(i<3):
        
        if(len(temp_input)>100):
            
            x_input=np.array(temp_input[1:])
            
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            
            temp_input.extend(yhat[0].tolist())
            
            lst_output.extend(yhat.tolist())
            i=i+1
    
    day_new=np.arange(1,101) #testdata 100indexes
    day_pred=np.arange(101,104) #101-131-predicted 

    plt.plot(day_new,scaler.inverse_transform(df1[1131:]))
    plt.plot(day_pred,scaler.inverse_transform(lst_output))
    plt.savefig('static/images/IDEA.png')
    


def display(start,end,symbol):
    df = pdr.get_data_yahoo(symbol, start,end)
    if symbol.upper() == 'AMZN':
        df.to_csv('AMZN.csv')
    elif symbol.upper() == 'TSLA':
        df.to_csv('TSLA.csv')
    elif symbol.upper() == 'GOOG':
        df.to_csv('GOOG.csv')
    elif symbol.upper() == 'AAPL':
        df.to_csv('AAPL.csv')
    elif symbol.upper() == 'IDEA.NS':
        df.to_csv('IDEANS.csv')

    df = get_data(symbol,start,end) 

    company_name = get_company_name(symbol.upper())
    fig= px.line(df,x=None,y='Close',title=company_name+' Close Price')
    fig.write_html("static/graph.html")
    fig2=px.line(df,x=None,y='Volume',title=company_name+' Volume')
    fig2.write_html('static/volume.html')
    #if(symbol=='IDEA.NS'):
    	#predictIdea()
    

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route("/sid/" , methods=["post"])
def hello():
    start=request.form.get('Strt_dt')
    end=request.form.get('End_dt')
    symbol=request.form.get('Name')
    
    display(start,end,symbol)
    return render_template('main.html')

@app.route("/prediction/Idea")
def predictedIdea():
    predictIdea()
    import time
    return render_template('Idea_Prediction_Graph.html')


@app.route('/news')
def news():
    newsapi = NewsApiClient(api_key="0144a0f2461949b7b8896463a90399f3")
    topheadlines = newsapi.get_top_headlines(sources="bbc-news")
 
    articles = topheadlines['articles']
 
    desc = []
    news = []
    img = []
 
    for i in range(len(articles)):
        myarticles = articles[i]
 
        news.append(myarticles['title'])
        desc.append(myarticles['description'])
        img.append(myarticles['urlToImage'])
 
    mylist = zip(news, desc, img)
 
    return render_template('bbc.html', context=mylist)

@app.route('/search')
def search():
    return render_template('search.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/dashboard')
def dashboard():
    return  render_template('Dashboard2.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return  render_template('create-acc.html')

@app.route('/snapshot')
def snapshot():
    with open('datasets/companies.csv') as f:
        for line in f:
            if "," not in line:
                continue
            symbol = line.split(",")[0] + '.NS'
            print("Symbol")
            print(symbol)
            data = yf.download(symbol, start="2020-01-01", end="2021-03-01")
            data.to_csv('datasets/daily/{}.csv'.format(symbol))

    return {
        "code": "success"
    }

@app.route('/screener')
def screener():
    pattern  = request.args.get('pattern', False)
    stocks = {}

    with open('datasets/companies.csv') as f:
        for row in csv.reader(f):
            stocks[row[0]] = {'company': row[1]}

    if pattern:
        for filename in os.listdir('datasets/daily'):
            df = pd.read_csv('datasets/daily/{}'.format(filename))
            pattern_function = getattr(talib, pattern)
            symbol = filename.split('.')[0]

            try:
                results = pattern_function(df['Open'], df['High'], df['Low'], df['Close'])
                last = results.tail(1).values[0]

                if last > 0:
                    stocks[symbol][pattern] = 'bullish'
                elif last < 0:
                    stocks[symbol][pattern] = 'bearish'
                else:
                    stocks[symbol][pattern] = None
            except Exception as e:
                print('failed on filename: ', filename)

    return render_template('screener.html', candlestick_patterns=candlestick_patterns, stocks=stocks, pattern=pattern)
   
if __name__ == '__main__':
    app.run(debug=True)



