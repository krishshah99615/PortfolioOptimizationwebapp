from flask import Flask, request, render_template
import numpy as np
import datetime
from final_port_opt import opt_data

app = Flask(__name__)


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

                start = '2013-01-01'
                end = datetime.date.today().strftime("%Y-%m-%d")
                amt = int(request.form['amt'])
                ticker = [c1n, c2n, c3n]
                weight = [c1, c2, c3]
                ans = opt_data(ticker, start, end, amt)
                print(ans)
                return render_template('result.html', d=ans)
            else:
                return render_template('index.html', err="Choose Unique Company")

        else:
            return render_template('index.html', err="All should add upto 10")
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
