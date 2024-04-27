import pickle
import pandas as pd
import streamlit as st

model = pickle.load(open("model.pkl","rb"))

def load_test_data(id):
    df = pd.read_csv("test_data.csv")
    sample = df.loc[df['SK_ID_CURR']==id]
    sample = sample.drop(columns=['SK_ID_CURR'])
    return sample

def predict(sample):
    proba = model.predict_proba(sample)[:, 1][0]
    seuil=0.575
    if proba >= seuil:
        Classe = "Accepté"
    else:
        Classe = "Refusé"
    return {'probabilite': proba, 'Classe': Classe}

def main():
    st.title("Prediction")
    html_temp = """
    <div style="background-color:green;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Prediction App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    id = st.text_input("Enter ID:", "")
    if id:
        sample = load_test_data(int(id))
        if not sample.empty:
            result = predict(sample)
            st.success(f'The output is: \nProbability: {result["probabilite"]} \nClass: {result["Classe"]}')
        else:
            st.error("The provided ID does not exist in the test dataset.")

if __name__ == '__main__':
    main()