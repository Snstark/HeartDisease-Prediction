import pickle
import numpy as np
import pandas as pd
import streamlit as st
import sklearn

st.title("Heart Disease Prediction")
data = pd.read_csv('data//heart_2020_cleaned.csv')
LogReg = pickle.load(open('Logreg.pkl', 'rb'))

nav = st.sidebar.radio('Navigation', ['Home', 'Data', 'Prediction', ])
if nav == "Data":
    st.dataframe(data)

if nav == "Home":
    st.image("data//download.jpeg")
    st.subheader("Heart Disease Prediction ")
    st.write('Prediction of person having a heart disease or not by calculating various parameter such as BMI,Age,etc.')
    st.write("""This dataset is Heart Disease Rate Prediction Data.
Already a Well CLeaned Dataset.
Data Set : taken from Kaggle.
According to the CDC, heart disease is one of the leading causes of death for people of most races in the US 
(African Americans, American Indians and Alaska Natives, and white people). About half of all Americans (47%) have 
at least 1 of 3 key risk factors for heart disease: high blood pressure, high cholesterol, and smoking. 
Other key indicator include diabetic status, obesity (high BMI), not getting enough 
physical activity or drinking too much alcohol. Detecting and preventing the factors that have 
the greatest impact on heart disease is very important in healthcare. Computational developments, in turn, 
allow the application of machine learning methods to detect "patterns" from the data that can predict a patient's condition.
  
HeartDisease : Respondents that have ever reported having coronary heart disease (CHD) or myocardial infarction (MI).
BMI : Body Mass Index (BMI).

Smoking : Have you smoked at least 100 cigarettes in your entire life? ( The answer Yes or No ).

AlcoholDrinking : Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week

Stroke : (Ever told) (you had) a stroke?

PhysicalHealth : Now thinking about your physical health, which includes physical illness and injury, 
                  for how many days during the past 30 days was your physical health not good? (0-30 days).

MentalHealth : Thinking about your mental health, for how many days during the past 30 days was your mental health not good? (0-30 days).

DiffWalking : Do you have serious difficulty walking or climbing stairs?

Sex : Are you male or female?

AgeCategory: Fourteen-level age category.

Race : Imputed race/ethnicity value.

Diabetic : (Ever told) (you had) diabetes?

PhysicalActivity : Adults who reported doing physical activity or exercise during the past 30 days other than their regular job.

GenHealth : Would you say that in general your health is...

SleepTime : On average, how many hours of sleep do you get in a 24-hour period?

Asthma : (Ever told) (you had) asthma?

KidneyDisease : Not including kidney stones, bladder infection or incontinence, were you ever told you had kidney disease?

SkinCancer : (Ever told) (you had) skin cancer?""")

if nav == "Prediction":
    st.subheader("Please enter following values")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        BMI = st.number_input("Enter BMI value", max_value=100.0)
        Smoking = st.selectbox("Do you Smoke?", ("Yes", "No"))
        AlcoholDrinking = st.selectbox("Do you drink Alcohol", ("Yes", "No"))
        Stroke = st.selectbox("Are you affected by stroke", ("Yes", "No"))
        Diabetic = st.radio("Are you diabetic",
                            ("No", "Yes", "No, borderline diabetes", "Yes (during pregnancy)(for female)"))
        PhysicalHealth = st.number_input("""Enter your physical Health in No. of days doing physical activity or exercise 
                             during the past 30 days other than your regular job""", max_value=30)

    with col2:
        MentalHealth = st.number_input("""Enter mental health, for how many days during the past 30 days
                                      was your mental health not good?""", max_value=30)
        DiffWalking = st.selectbox("Do you suffer any serious difficulty walking or climbing stairs?", ("Yes", "No"))
        PhysicalActivity = st.selectbox("Do you indulge in physical activity or exercise during "
                                        "the past 30 days other than their regular job.", ("Yes", "No"))
        GenHealth = st.radio("Rate the your general health",
                             ("Very good", "Good", "Excellent", "Fair", "Poor"))
        SleepTime = st.number_input("On average, how many hours of sleep you get in a 24-hour period?",
                                    min_value=1, max_value=24)

    with col3:
        Sex = st.selectbox("Please mention the gender", ("Male", "Female"))
        AgeCategory = st.radio("Select the Age group from below",
                               ("80 or older", "75-79", "70-74", "65-69", "60-64", "55-59", "50-54",
                                "45-49", "40-44", "35-39", "30-34", "25-29", "18-24"))
        Asthma = st.selectbox("Does person have Asthma", ("Yes", "No"))
        KidneyDisease = st.selectbox("Do you suffer from any  Kidney Disease", ("Yes", "No"))
        SkinCancer = st.selectbox("Do you suffer from any Skin Cancer Disease", ("Yes", "No"))

    if Sex == "Male":
        Sex = 1
    elif Sex == "Female":
        Sex = 0

    if Asthma == "Yes":
        Asthma = 1
    elif Asthma == "No":
        Asthma = 0

    if KidneyDisease == "Yes":
        KidneyDisease = 1
    elif KidneyDisease == "No":
        KidneyDisease = 0

    if SkinCancer == "Yes":
        SkinCancer = 1
    elif SkinCancer == "No":
        SkinCancer = 0

    if AgeCategory == "65-69":
        AgeCategory = 1
    elif AgeCategory == "60-64":
        AgeCategory = 2
    elif AgeCategory == "70-74":
        AgeCategory = 3
    elif AgeCategory == "55-59":
        AgeCategory = 4
    elif AgeCategory == "50-54":
        AgeCategory = 5
    elif AgeCategory == "80 or older":
        AgeCategory = 6
    elif AgeCategory == "45-49":
        AgeCategory = 7
    elif AgeCategory == "75-79":
        AgeCategory = 8
    elif AgeCategory == "18-24":
        AgeCategory = 9
    elif AgeCategory == "40-44":
        AgeCategory = 10
    elif AgeCategory == "35-39:":
        AgeCategory = 11
    elif AgeCategory == "30-34":
        AgeCategory = 12
    elif AgeCategory == "25-29":
        AgeCategory = 13

    if Smoking == "Yes":
        Smoking = 1
    elif Smoking == "No":
        Smoking = 0

    if AlcoholDrinking == "Yes":
        AlcoholDrinking = 1
    elif AlcoholDrinking == "No":
        AlcoholDrinking = 0

    if Stroke == "Yes":
        Stroke = 1
    elif Stroke == "No":
        Stroke = 0

    if Diabetic == "No":
        Diabetic = 2
    elif Diabetic == "Yes":
        Diabetic = 1
    elif Diabetic == "No, borderline diabetes":
        Diabetic = 4
    elif Diabetic == "Yes (during pregnancy)(for female)":
        Diabetic = 3

    if DiffWalking == "Yes":
        DiffWalking = 1
    elif DiffWalking == "No":
        DiffWalking = 0

    if PhysicalActivity == "Yes":
        PhysicalActivity = 1
    elif PhysicalActivity == "No":
        PhysicalActivity = 0

    if GenHealth == "Very good":
        GenHealth = 5
    elif GenHealth == "Good":
        GenHealth = 4
    elif GenHealth == "Excellent":
        GenHealth = 3
    elif GenHealth == "Fair":
        GenHealth = 2
    elif GenHealth == "Poor":
        GenHealth = 1

    x = np.array(
        [BMI, Smoking, AlcoholDrinking, Stroke, PhysicalHealth, MentalHealth, DiffWalking, Sex, AgeCategory, Diabetic,
         PhysicalActivity, GenHealth, SleepTime, Asthma, KidneyDisease, SkinCancer])
    x = x.reshape(1, 16)

    if st.button("Predict"):
        S = LogReg.predict(x)
        if S == 1:
            st.header("You have High risk of getting Heart Diseases")
        else:
            st.header("You have Lesser Chance of getting Heart Diseases")
