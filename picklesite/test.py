#!/usr/bin/python3.6
import sys
import os
from flask import send_from_directory
print(send_from_directory)
import time
from flask import Flask, render_template, flash, redirect, url_for, session, logging, request
#from data import Articles
import mysql.connector
from wtforms import Form, StringField, TextAreaField, PasswordField, validators
from werkzeug import generate_password_hash
#from passlib.hash import sha256_crypt

from functools import wraps
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as mpld3
import pandas as pd
import matplotlib.dates as dates
from matplotlib import ticker
import MySQLdb
import matplotlib.animation as animation
import matplotlib.animation as manimation; manimation.writers.list()
#import mpl_finance



app = Flask(__name__)
#rough attempt
cnx = mysql.connector.connect(user='mar0093', password='pickledb',
                              host='mar0093.mysql.pythonanywhere-services.com',
                              database='mar0093$pickledb_pyany')
cnx.close()
'''
#configuring MySQL
app.config['MYSQL_HOST'] = 'mar0093.mysql.pythonanywhere-services.com'
app.config['MYSQL_USER'] = 'mar0093'
app.config['MYSQL_PASSWORD'] = 'pickledb'
app.config['MYSQL_DB'] = 'mar0093$pickledb_pyany'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
#init MySQL
mysql = MYSQL(app)
'''

## Below pulls articles from data.py##
#Articles = Articles()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/watchlist')
def shares():
    cur = mysql.connection.cursor()
    # Get articles
    result = cur.execute("SELECT * FROM articles")
    # Get all articles
    articles = cur.fetchall()
    cur.close()
    if result > 0:
        return render_template('watchlist.html', articles=articles)
    else:
        flash('No Articles Found.', 'danger')
        return render_template('watchlist.html')
    # CLose connection

@app.route('/shares/<string:id>/')
def shares_disp(id):
    cur = mysql.connection.cursor()
    # Get articles
    result = cur.execute("SELECT * FROM articles WHERE id = %s", [id])
    # Get all articles
    article = cur.fetchone()
    return render_template('share_disp.html', article=article)

@app.route('/shares2/<string:aticker>/<string:graph_type>',methods=['GET', 'POST'])
def shares_disp2(aticker, graph_type):
    print("a")
    cur = mysql.connection.cursor()
    # Get articles
    cap_id = aticker.upper()
    result = cur.execute("SELECT * FROM " + aticker)
    # Get all articles
    my_result = cur.execute("SELECT * FROM " + aticker + " ORDER BY Date DESC LIMIT 1")
    my_result = cur.fetchone()
    cur.close()
    cur = mysql.connection.cursor()
    print('b')
    sql = "SELECT * FROM " + aticker
    cur.execute(sql)
    all_result = cur.fetchall()
    df = pd.DataFrame(list(all_result), columns=["Date", "Open", "Close", "High", "Low", "Volume"])
    df["Close_av_100"] = df['Close'].rolling(100).mean()
    df["Close_av_30"] = df['Close'].rolling(30).mean()
    if graph_type =="default":
        df = df.tail(60)
    else:
        df = df.tail(int(graph_type))
    print("c")
    df["mpld3_date"] =[dt.datetime.strptime(d, '%Y-%m-%d') for d in df["Date"]]
    df["Date"] = [dt.datetime.strptime(d, '%Y-%m-%d').toordinal() for d in df["Date"]]

    df = df.set_index(df["Date"])
    quotes = [tuple([df["Date"].tolist()[i],
                     df["Open"].tolist()[i],
                     df["High"].tolist()[i],
                     df["Low"].tolist()[i],
                     df["Close"].tolist()[i]]) for i in range(len(df.index))]  # _1

    fig, [ax, ax2] = plt.subplots(2, 1,figsize=(12, 7), sharex=True)
    ax = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=2, colspan=1,sharex=ax)
    print("d")
    mpl_finance.candlestick_ohlc(ax, quotes,  colorup='g', colordown='r',width=0.6)
    print("e")
    ax.plot(df["Date"].tolist(), df["Close_av_100"],label='MA_100')
    ax.plot(df["Date"].tolist(), df["Close_av_30"],label='MA_30')
    candle_colours = []
    for i in range(len(df["Date"].tolist())):
        if df["Open"].tolist()[i] >= df["Close"].tolist()[i]:
            candle_colours.append('r')
        else:
            candle_colours.append('g')
    ax2.bar(df["Date"].tolist(), df["Volume"], color=candle_colours)
    ax.legend()
    ax.xaxis_date()
    ax2.xaxis_date()
    print("f")
    fig.autofmt_xdate()
    fig.tight_layout()
    end = int(len(df["Open"])-1)
    pct_change = round(((df['Open'].iloc[end]/df['Open'].iloc[0])-1)*100,2)
    test_var = mpld3.fig_to_html(fig)
    print('here')
    #test_var2 = mpld3.fig_to_html(fig)
    cur.close()
    return render_template('share_disp2.html', test_var=test_var, id=cap_id, result=my_result, pct_change=pct_change)




class RegisterForm(Form):
    name = StringField('Name', [validators.Length(min=1, max=50)])
    username = StringField('Username', [validators.Length(min=4, max=25)])
    email = StringField('Email', [validators.Length(min=6, max=50)])
    password = PasswordField('Password', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message='Passwords do not match')
    ])
    confirm = PasswordField('Confirm Password')

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm(request.form)
    if request.method == 'POST' and form.validate():
        name = form.name.data
        email = form.email.data
        username = form.username.data
        password = sha256_crypt.encrypt(str(form.password.data)) ###################################################################

        #create cursor
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users(name, email, username, password) VALUES(%s, %s, %s, %s)",
        (name, email, username, password))

        #commit to DB
        mysql.connection.commit()

        #close connection
        cur.close()

        flash('You are now registered and can log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

#user login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        #Get form fields
        username = request.form['username']
        password_canidate = request.form['password']

        # Create cursor
        cur = mysql.connection.cursor()

        # get user by username
        result = cur.execute("SELECT * FROM users WHERE username = %s", [username])
        if result > 0:
            #Get stored hash
            data = cur.fetchone()
            password = data['password']

            #Compare passwords.
            if sha256_crypt.verify(password_canidate, password):
                app.logger.info('PASSWORD MATCHED')
                session['logged_in'] = True
                session['username'] = username
                flash('You are now logged in.', 'success')
                return redirect(url_for('dashboard'))
            else:
                app.logger.info('WRONG PASSWORD')
            flash('Incorrect Password.', 'danger')
            #Close current session
            cur.close()
        else:
            app.logger.info('NO USER')
            flash('Unknown Username', 'danger')
    return render_template('login.html')

#check if user logged in
def is_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Unauthorized, Please login', 'danger')
            return redirect(url_for('login'))
    return wrap

@app.route('/logout')
@is_logged_in
def logout():
    session.clear()
    flash('You are now logged out.', 'success')
    return redirect(url_for('login'))

#user home page
@app.route('/dashboard')
@is_logged_in
def dashboard():
    cur = mysql.connection.cursor()
    # Get articles
    result = cur.execute("SELECT * FROM articles")
    # Get all articles
    articles = cur.fetchall()
    cur.close()
    if result > 0:
        return render_template('dashboard.html', articles=articles)
    else:
        flash('No Articles Found.', 'danger')
        return render_template('dashboard.html')
    # CLose connection

#user home page
@app.route('/scout')
@is_logged_in
def scout():
    cur = mysql.connection.cursor()
    # Get articles
    result = cur.execute("SELECT * FROM watchlist where User = '"+session['username']+"'")
    # Get all articles
    watchlist = cur.fetchall()
    cur.close()
    if result > 0:
        return render_template('scout.html', watchlist=watchlist)
    else:
        flash('Nothing in Watchlist', 'danger')
        return render_template('scout.html')
    # CLose connection




@app.route('/add_scout', methods=['GET', 'POST'])
@is_logged_in
def add_scout():
    """Only need the ticker name, can get the rest from sql and session."""
    print("into scout")
    if request.method == 'POST':
        print("post recognised")

        ticker = request.form.get('add')
        ticker = ticker.lower()
        print('ticker is ' + str(ticker))
        #create Cursor
        cur = mysql.connection.cursor()

        #Execute
        #result = cur.execute("SELECT * FROM " + ticker.lower())
        other_result = cur.execute("SELECT * FROM "+ ticker + " ORDER BY Date DESC LIMIT 1")
        all_result = cur.fetchall()
        df = pd.DataFrame(list(all_result), columns=["Date", "Open", "Close", "High", "Low", "Adj_Close", "Volume"])

        # prevent duplicate added to watchlist
        watchlist_result = cur.execute("SELECT * FROM watchlist")
        all_result2 = cur.fetchall()
        df2 = pd.DataFrame(list(all_result2), columns=["User","Ticker"])

        duplicate_count = False
        for i in range(len(df2)):
            if df2["User"].iloc[i] == session['username'] and df2["Ticker"].iloc[i] == ticker:
                duplicate_count = True
                break
        if duplicate_count:
            flash('Already in scout', 'danger')
        else:
            cur.execute(
                "INSERT INTO watchlist(Ticker, Watchlist_title, User, Open, High, Low, Close, Adj_Close, Volume) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (ticker, 1, session['username'], df['Open'].iloc[0], df['High'].iloc[0], df['Low'].iloc[0], df['Close'].iloc[0], df['Adj_Close'].iloc[0], df['Volume'].iloc[0]))
            #commit to db
            mysql.connection.commit()

            #close connection
            cur.close()

            flash('Added to scout', 'success')


    return render_template('index.html')

#article form class
class ArticleForm(Form):
    title = StringField('Title', [validators.Length(min=1, max=200)])
    body = TextAreaField('Body', [validators.Length(min=30)])

@app.route('/add_article', methods=['GET', 'POST'])
@is_logged_in
def add_article():
    form = ArticleForm(request.form)
    if request.method == 'POST' and form.validate():
        title = form.title.data
        body = form.body.data

        #create Cursor
        cur = mysql.connection.cursor()

        #Execute
        cur.execute("INSERT INTO articles(title, body, author) VALUES(%s, %s, %s)", (title, body, session['username']))

        #commit to db
        mysql.connection.commit()

        #close connection
        cur.close()

        flash('Article Created', 'success')

        return redirect(url_for('dashboard'))
    return render_template('add_article.html', form=form)


@app.route('/edit_article/<string:id>', methods=['GET', 'POST'])
@is_logged_in
def edit_article(id):
    # Create cursor
    cur = mysql.connection.cursor()

    #Get article by id
    result = cur.execute("SELECT * FROM articles WHERE id = %s", [id])

    article = cur.fetchone()
    #get form
    form = ArticleForm(request.form)

    #populate article form fields (title & body)
    form.title.data = article['title']
    form.body.data = article['body']
    if request.method == 'POST' and form.validate():
        title = request.form['title']
        body = request.form['body']

        #create Cursor
        cur = mysql.connection.cursor()

        #Execute
        cur.execute("UPDATE articles SET title=%s, body=%s WHERE id = %s", (title, body, id))

        #commit to db
        mysql.connection.commit()

        #close connection
        cur.close()

        flash('Article Created', 'success')

        return redirect(url_for('dashboard'))
    return render_template('edit_article.html', form=form)

#Delete article
@app.route('/delete_article/<string:id>', methods=['POST'])
@is_logged_in
def delete_article(id):
    # create cursor
    cur = mysql.connection.cursor()

    #execute
    cur.execute("DELETE FROM articles WHERE id = %s", [id])

    #Commit to db
    mysql.connection.commit()

    #Close connection
    cur.close()
    flash('Article deleted.', 'success')

    return redirect(url_for('dashboard'))

#Delete row
@app.route('/delete_row/<string:id>', methods=['POST'])
@is_logged_in
def delete_row(id):
    # create cursor
    cur = mysql.connection.cursor()

    #execute
    cur.execute("DELETE FROM watchlist WHERE id = %s", [id])

    #Commit to db
    mysql.connection.commit()

    #Close connection
    cur.close()
    flash('Share removed.', 'success')

    return redirect(url_for('scout'))

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico')

if __name__ == '__main__':

    app.secret_key = 'my_pickle'
    app.run()