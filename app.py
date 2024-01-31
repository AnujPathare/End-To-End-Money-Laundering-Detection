from flask import Flask, request, app,render_template

from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

## Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

## Route for prediction
@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='GET':
        return render_template('index.html')
    else:
        data = CustomData(
            From_Bank=request.form.get('From_Bank'),
            Sender_Account=request.form.get('Sender_Account'),
            To_Bank=request.form.get('To_Bank'),
            Receiver_Account=request.form.get('Receiver_Account'),
            Amount_Received=request.form.get('Amount_Received'),
            Receiving_Currency=request.form.get('Receiving_Currency'),
            Amount_Paid=request.form.get('Amount_Paid'),
            Payment_Currency=request.form.get('Payment_Currency'),
            Payment_Format=request.form.get('Payment_Format'),
            Day=request.form.get('Day'),
            Hour=request.form.get('Hour'),
            Minute=request.form.get('Minute')
        )

        pred_df = data.get_data_as_dataframe()

        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        results = ['Legitimate' if results[0] == 0 else 'Money Laundering']
        return render_template('index.html',results=results[0])


if __name__=="__main__":
    app.run()