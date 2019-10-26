
from flask import send_from_directory
import time
from flask import Flask, render_template, flash, redirect, url_for, session, logging, request
#from data import Articles
from flask_mysqldb import MySQL
import pandas as pd
app = Flask(__name__)

#configuring MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'pickle'
app.config['MYSQL_DB'] = 'pickledb'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
#init MySQL
mysql = MySQL(app)

def top_20():
    """
    traverse through all tickers and find biggest change [day, week, month]
    :return:
    """
    with open('tickers.txt') as f: #need actual directory for main copy
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [x[:3] for x in content]
    print(content)
    cur = mysql.connection.cursor()
    for aticker in content: # grab the ticker a get it's week month day pct change.

        cap_id = aticker.upper()
        result = cur.execute("SELECT * FROM " + aticker)
        my_result = cur.execute("SELECT * FROM " + aticker + " ORDER BY Date DESC LIMIT 1")
        my_result = cur.fetchone()
        cur.close()
        cur = mysql.connection.cursor()
        sql = "SELECT * FROM " + aticker
        cur.execute(sql)
        all_result = cur.fetchall()
        AI_df = pd.DataFrame(list(all_result), columns=['Close', 'Open', 'High', 'Low', 'Volume'])


top_20()
