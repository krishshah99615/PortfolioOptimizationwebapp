from flask import Flask, request, render_template
import numpy as np
import datetime
from final_port_opt import opt_data
from dateutil.relativedelta import relativedelta
from pred import get_test_graph, pred_next
import glob
import tensorflow as tf

app = Flask(__name__)
f = open("tickerlist.txt", 'r')
t_list = dict()

for line in f:
    t_list[line.split()[1]] = line.split()[0]


@app.route("/", methods=['GET', 'POST'])
def index():

    if request.method == "POST":

        c1 = int(request.form['c1'])/10
        c3 = int(request.form['c3'])/10
        c2 = int(request.form['c2'])/10

        if((c1+c2+c3) == 1.0):
            c1n = request.form['c1n']
            c2n = request.form['c2n']
            c3n = request.form['c3n']
            if len(list(set([c1n, c2n, c3n]))) == 3:
                start = (datetime.datetime.now() +
                         relativedelta(years=-10)).strftime("%Y-%m-%d")
                end = datetime.datetime.now().strftime("%Y-%m-%d")
                '''start = '2013-01-01'
                end = datetime.date.today().strftime("%Y-%m-%d")'''
                amt = int(request.form['amt'])
                ticker = [c1n, c2n, c3n]
                print(ticker)
                weight = [c1, c2, c3]

                ans = opt_data(ticker, start, end, amt)

                print("Loading models...")
                models = {}
                for t in ticker:
                    p = glob.glob(f'*{t}.h5')[0]
                    print(f"Loading..{p}")
                    models[t] = tf.keras.models.load_model(p)

                o = []
                for tic in ticker:

                    predicted_stock_price, real_val, sc = get_test_graph(
                        tic, models[tic])
                    o.append(pred_next(models[tic], sc))

                pred_list = []
                for pred_df in o:
                    a = pred_df.values.reshape((-1))
                    i = list(pred_df.index)
                    i = [x.strftime("%Y-%m-%d") for x in i]
                    pred_list.append([d for d in zip(i, a)])

                return render_template('result.html', d=ans, o=pred_list)
            else:
                return render_template('index.html', err="Choose Unique Company", t=t_list)

        else:
            return render_template('index.html', err="All should add upto 10", t=t_list)
    return render_template('index.html', t=t_list)


if __name__ == "__main__":
    app.run(debug=True)
