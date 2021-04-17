from flask import Flask, request,render_template
import pandas as pd
from werkzeug.utils import secure_filename
import numpy as np 
from statsmodels.tsa.arima_model import ARIMA
import os

app = Flask(__name__)

@app.route('/')
def form():
    return """
        <html>
            <body>
                <h1 align=center>COVID19 PREDICTIONS</h1>

                <form action="/transform" method="post" enctype="multipart/form-data">
                    <table><tr><td>
                    Upload the current covid data:</td>
                    <td><!--<input type="file" name="data_file" />-->
                    </td>
                    </tr>
                    <tr>
                    <td>
                    Number of days to Predict: </td><td><input type="text" value="90" name="predict_days" />
                     </td>
                    </tr>
                    <tr>
                    <td colspan=2 align=right>
                    <input type="submit" value = "Predict"/>
                    </td>
                    </tr>
                </form>

            </body>
        </html>
    """

@app.route('/transform', methods=["POST"])
def transform_view():
    if request.method == 'POST':
        numdays= request.form['predict_days']
        
        # data_file is the name of the file upload field
        #f = request.files['data_file']
        # for security - stops a hacker e.g. trying to overwrite system files
        #filename = secure_filename(f.filename)
        # save a copy of the uploaded file
        #f.save(filename)
        filename='data/Data.csv'
        df = pd.read_csv(filename, index_col='Date', parse_dates=True)
        df_Confirmed  = df.drop(['Recovered','Deaths'],axis=1)
        df_Recovered  = df.drop(['Confirmed','Deaths'],axis=1)
        df_Deaths  = df.drop(['Confirmed','Recovered'],axis=1)
        
        def train(dataset, p,d,q):
            model = ARIMA(dataset, order=(p, d, q)) 
            results_ARIMA = model.fit()
            Errors = results_ARIMA.resid
            print(np.sqrt(np.mean((Errors)**2)))      
            return results_ARIMA

        Confirmed_path='static/images/df_Confirmed.png'
        Recovered_path='static/images/df_Recovered.png'
        Deaths_path='static/images/df_Deaths.png'
        
        if (os.path.isfile(Confirmed_path)):
            os.remove(Confirmed_path)
            
        if (os.path.isfile(Recovered_path)):
            os.remove(Recovered_path)
        
        if (os.path.isfile(Deaths_path)):
            os.remove(Deaths_path)
        
        start_index = int(len(df)*70/100)
        stop_index = len(df) + int(numdays)
        train(df_Confirmed, 10, 1, 2).plot_predict(start_index,stop_index).savefig(Confirmed_path)
        train(df_Recovered, 10, 1, 2).plot_predict(start_index,stop_index).savefig(Recovered_path)
        train(df_Deaths, 10, 1, 2).plot_predict(start_index,stop_index).savefig(Deaths_path)

        return render_template('index.html',  user_image = 'df_Confirmed.png', user_image2 = 'df_Recovered.png', user_image3 = 'df_Deaths.png')
        
        
    return 'Oops, Try again something went wrong!'

if __name__ == '__main__':
    app.run(debug=True)
