import os
import pickle
import pandas as pd
import streamlit as st

#from flask import Flask, jsonify, request
#app = Flask(__name__)
#model_path = os.path.join(current_directory, "..", "Simulations", "Best_model", "model.pkl")
model = pickle.load(open("model.pkl","rb"))
def predict():
    id =int(request.args.get('id'))
    if id==None :
        return "cet identifiant n'existe pas"
    else:
        print(str(id))
        df = pd.read_csv("test_data.csv")
        sample = df.loc[df['SK_ID_CURR']==id]
        #print(sample)
        sample = sample.drop(columns=['SK_ID_CURR'])
        #print(sample)
        proba = model.predict_proba(sample)[:, 1][0]
        #proba = prediction[0][1]
        print(proba)
        seuil=0.575
        if proba >= seuil:
            Classe = "Accepté"
        else:
            Classe = "Refusé"
        return jsonify({'probabilite': proba, 'Classe': Classe})
    
def main():
    st.title("prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    id = st.text_input("id","Type Here")
    result=""
    if st.button("Predict"):
        result=predict()
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")
if __name__ == '__main__':
    main()

