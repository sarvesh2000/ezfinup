from flask import Flask, render_template,request, redirect, session, jsonify
import os, csv
from newsapi import NewsApiClient
import talib
import yfinance as yf
from patterns import candlestick_patterns
import firebase_admin
from firebase_admin import credentials, auth
import pyrebase
import json
import pandas as pd

# Firebase Admin Init
cred = credentials.Certificate('fbAdminConfig.json')
firebase_admin = firebase_admin.initialize_app(cred)

# Firebase Init
firebase = pyrebase.initialize_app(json.load(open('fbConfig.json')))


# Init
newsapi = NewsApiClient(api_key='0144a0f2461949b7b8896463a90399f3')


key="483bd89d87mshce534f1bd4ba6b0p182fcdjsnf08d8fc19c04"

# App Begins
app = Flask(__name__)
app.secret_key = 'the random string'

@app.route('/')
def home():
    return render_template('index.html')

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

@app.route('/dashboard')
def dashboard():
    return  render_template('madhu_dashboard.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        email = request.form.get('email')
        password = request.form.get('password')
        # try:
        user = firebase.auth().sign_in_with_email_and_password(email, password)
        jwt = user['idToken']
        session['user_id'] = jwt
        return redirect('/dashboard')
        # except:
        #     return {'message': 'There was an error logging in'},400
    else:
        return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("cnf-password")
        if password == confirm_password:
            # try:
            user = auth.create_user(
                email=email,
                password=password
            )
            # session['user_id'] = user.idToken
            # return redirect('/dashboard')
            return jsonify(user)
            # except:
            #     return {'message': 'Error creating user'},400
        else:
            return {'message': 'Error creating user. Password and Confirm Password don\'t match'},400
    else:
        return  render_template('create-acc.html')

# Candlestick routes
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
    return  render_template('create-acc.html')

# End of candlestick routes

@app.route('/aboutpage')
def aboutpage():
    return render_template('aboutpage.html')

@app.route('/home')
def homepage():
    return render_template('homepage.html')
    
   
if __name__ == '__main__':
    app.run(debug=True)



