import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import pandas as pd


try:
    grid_search = pickle.load(open('model_fraud.sav', 'rb'))
except FileNotFoundError:
    st.error("Model file Tidak ada")
    st.stop()


if not isinstance(grid_search, GridSearchCV):
    st.error("model file alah")
    st.stop()

model_fraud = grid_search.best_estimator_


try:
    loaded_vec = TfidfVectorizer(decode_error="replace",
                                 vocabulary=set(pickle.load(open("new_selected_feature_tf-idf.sav", "rb"))))
except FileNotFoundError:
    st.error("Vocabulary file tidak ada")
    st.stop()



try:
    data = pd.read_csv('Data/clean_data.csv') 
except FileNotFoundError:
    st.error("file tidak ada")
    st.stop()

data['clean_teks'].fillna('', inplace=True)
data['clean_teks'] = data['clean_teks'].astype(str)

if 'clean_teks' not in data.columns:
    st.error("The 'clean_teks' column is missing")
    st.stop()
    

loaded_vec.fit(data['clean_teks']) 


st.title('Klasifikasi SMS Penipuan MNB')
st.write(f"**Model Accuracy (Best Score):** {grid_search.best_score_:.2f}")

input_method = st.radio("Pilih input method", ["Text Input", "File Upload"])

if input_method == "Text Input":
    clean_teks = st.text_area('Masukan SMS')  
    input_texts = [clean_teks]
else:
    uploaded_files = st.file_uploader("Choose text files", type="txt", accept_multiple_files=True)
    input_texts = []
    for uploaded_file in uploaded_files:
        file_contents = uploaded_file.read().decode('utf-8')
        input_texts.extend(file_contents.split('\n'))

if st.button('Hasil'):
    if input_texts:
        prediction_results = []
        for text in input_texts:
            if text.strip():
                
                predict_fraud = model_fraud.predict(loaded_vec.transform([text]))
                if predict_fraud == 0:
                    fraud_detection = 'SMS Normal'
                elif predict_fraud == 1:
                    fraud_detection = 'SMS Fraud'
                else:
                    fraud_detection = 'SMS Promo'


                probability = model_fraud.predict_proba(loaded_vec.transform([text]))[0]
                prediction_results.append({
                    'Input': text, 
                    'Prediction': fraud_detection, 
                    'Probability (Normal)': f"{probability[0]:.2f}",
                    'Probability (Fraud)': f"{probability[1]:.2f}",
                    'Probability (Promo)': f"{probability[2]:.2f}"
                })

        if prediction_results:
            results_df = pd.DataFrame(prediction_results)
            st.table(results_df)
    else:
        st.warning('No input provided.')