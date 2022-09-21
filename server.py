from flask import Flask, render_template, request
import pickle
import pandas as pd 
from model import *

app = Flask(__name__)

recommendation_model = pickle.load(open('recommendation_system_model.pkl','rb'))
sentiment_analysis_model=pickle.load(open('sentiment_analysis_model.pkl','rb'))
tf_idf_vect_model=pickle.load(open('tf_idf_model.pkl','rb'))


@app.route("/",methods=['POST','GET'])
def recommend_product():
    if request.method == 'POST':
        user=request.form.get("username")
        recommend=recommendation_model.loc[user].sort_values(ascending=False)[0:20].index.tolist()
        df=pd.read_csv('sample30.csv')
        selected_product=df[df.name.isin(recommend)]
        word_features=tf_idf_vect_model.get_feature_names_out()
        reviews=pd.DataFrame([extract_features(sentence,word_features) for sentence in selected_product['reviews_text'].values])
        y_pred = sentiment_analysis_model.predict(reviews)
        output=pd.DataFrame(data={"product":selected_product['name'],"sentiment":y_pred})
        recommended_products=output[:5]
        print(len(output))
        return render_template("index.html",product_list=recommend,recommended_product_list=[recommended_products['name']])
    if request.method == 'GET':
        return render_template("index.html")

@app.route("/submit")
def submit():
    return "Hello from submit page"

if __name__ == '__main__':
    app.run()