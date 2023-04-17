#importing all the important libraries
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#building the sidebar of the web app which will help us navigate through the different sections of the entire application
rad=st.sidebar.radio("Navigation Menu",["Home","Covid-19","Diabetes","Heart Disease","Liver","Breast_cancer"])

#Home Page 

#displays all the available disease prediction options in the web app
if rad=="Home":
    st.title("Medical Predictions App")
    st.image("Medical Prediction Home Page.jpg")
    st.text("The Following Disease Predictions Are Available ->")
    st.text("1. Covid-19 Infection Predictions")
    st.text("2. Diabetes Predictions")
    st.text("3. Liver Disease Predictions")
    st.text("4. Kidney Disease Predictions")
    st.text("5. Breast Disease Predictions")
    # st.text("3. Heart Disease Predictions")

#Covid-19 Prediction

#loading the Covid-19 dataset
df1=pd.read_csv("Covid-19 Predictions.csv")
#cleaning the data by dropping unneccessary column and dividing the data as features(x1) & target(y1)
x1=df1.drop("Infected with Covid19",axis=1)
x1=np.array(x1)
y1=pd.DataFrame(df1["Infected with Covid19"])
y1=np.array(y1)
#performing train-test split on the data
x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.2,random_state=0)
#creating an object for the model for further usage
model1=RandomForestClassifier()
#fitting the model with train data (x1_train & y1_train)
model1.fit(x1_train,y1_train)

#Covid-19 Page

#heading over to the Covid-19 section
if rad=="Covid-19":
    st.header("Know If You Are Affected By Covid-19")
    st.write("All The Values Should Be In Range Mentioned")
    #taking the 4 most important features as input as features -> Dry Cough (drycough), Fever (fever), Sore Throat (sorethroat), Breathing Problem (breathingprob)
    #a min value (min_value) & max value (max_value) range is set so that user can enter value within that range
    #incase user enters a value which is not in the range then the value will not be taken whereas an alert message will pop up
    drycough=st.number_input("Rate Of Dry Cough (0-20)",min_value=0,max_value=20,step=1)
    fever=st.number_input("Rate Of Fever (0-20)",min_value=0,max_value=20,step=1)
    sorethroat=st.number_input("Rate Of Sore Throat (0-20)",min_value=0,max_value=20,step=1)
    breathingprob=st.number_input("Rate Of Breathing Problem (0-20)",min_value=0,max_value=20,step=1)
    #the variable prediction1 predicts by the health state by passing the 4 features to the model
    prediction1=model1.predict([[drycough,fever,sorethroat,breathingprob]])[0]
    
    #prediction part predicts whether the person is affected by Covid-19 or not by the help of features taken as input
    #on the basis of prediction the results are displayed
    if st.button("Predict"):
        if prediction1=="Yes":
            st.warning("You Might Be Affected By Covid-19")
        elif prediction1=="No":
            st.success("You Are Safe")

#Diabetes Prediction

#loading the Diabetes dataset
df2=pd.read_csv("Diabetes Predictions.csv")
#cleaning the data by dropping unneccessary column and dividing the data as features(x2) & target(y2)
x2=df2.iloc[:,[1,4,5,7]].values
x2=np.array(x2)
y2=y2=df2.iloc[:,[-1]].values
y2=np.array(y2)
#performing train-test split on the data
x2_train,x2_test,y2_train,y2_test=train_test_split(x2,y2,test_size=0.2,random_state=0)
#creating an object for the model for further usage
model2=RandomForestClassifier()
#fitting the model with train data (x2_train & y2_train)
model2.fit(x2_train,y2_train)

#Diabetes Page

#heading over to the Diabetes section
if rad=="Diabetes":
    st.header("Know If You Are Affected By Diabetes")
    st.write("All The Values Should Be In Range Mentioned")
    #taking the 4 most important features as input as features -> Glucose (glucose), Insulin (insulin), Body Mass Index-BMI (bmi), Age (age)
    #a min value (min_value) & max value (max_value) range is set so that user can enter value within that range
    #incase user enters a value which is not in the range then the value will not be taken whereas an alert message will pop up
    glucose=st.number_input("Enter Your Glucose Level (0-200)",min_value=0,max_value=200,step=1)
    insulin=st.number_input("Enter Your Insulin Level In Body (0-850)",min_value=0,max_value=850,step=1)
    bmi=st.number_input("Enter Your Body Mass Index/BMI Value (0-70)",min_value=0,max_value=70,step=1)
    age=st.number_input("Enter Your Age (20-80)",min_value=20,max_value=80,step=1)
    #the variable prediction1 predicts by the health state by passing the 4 features to the model
    prediction2=model2.predict([[glucose,insulin,bmi,age]])[0]
    
    #prediction part predicts whether the person is affected by Diabetes or not by the help of features taken as input
    #on the basis of prediction the results are displayed
    if st.button("Predict"):
        if prediction2==1:
            st.warning("You Might Be Affected By Diabetes")
        elif prediction2==0:
            st.success("You Are Safe")

#Heart Disease Prediction

#loading the Heart Disease dataset
df3=pd.read_csv("Heart Disease Predictions.csv")
#cleaning the data by dropping unneccessary column and dividing the data as features(x3) & target(y3)
x3=df3.iloc[:,[2,3,4,7]].values
x3=np.array(x3)
y3=y3=df3.iloc[:,[-1]].values
y3=np.array(y3)
#performing train-test split on the data
x3_train,x3_test,y3_train,y3_test=train_test_split(x3,y3,test_size=0.2,random_state=0)
#creating an object for the model for further usage
model3=RandomForestClassifier()
#fitting the model with train data (x3_train & y3_train)
model3.fit(x3_train,y3_train)

#Heart Disease Page

#heading over to the Heart Disease section
if rad=="Heart Disease":
    st.header("Know If You Are Affected By Heart Disease")
    st.write("All The Values Should Be In Range Mentioned")
    #taking the 4 most important features as input as features -> Chest Pain (chestpain), Blood Pressure-BP (bp), Cholestrol (cholestrol), Maximum HR (maxhr)
    #a min value (min_value) & max value (max_value) range is set so that user can enter value within that range
    #incase user enters a value which is not in the range then the value will not be taken whereas an alert message will pop up
    chestpain=st.number_input("Rate Your Chest Pain (1-4)",min_value=1,max_value=4,step=1)
    bp=st.number_input("Enter Your Blood Pressure Rate (95-200)",min_value=95,max_value=200,step=1)
    cholestrol=st.number_input("Enter Your Cholestrol Level Value (125-565)",min_value=125,max_value=565,step=1)
    maxhr=st.number_input("Enter You Maximum Heart Rate (70-200)",min_value=70,max_value=200,step=1)
    #the variable prediction1 predicts by the health state by passing the 4 features to the model
    prediction3=model3.predict([[chestpain,bp,cholestrol,maxhr]])[0]
    
    #prediction part predicts whether the person is affected by Heart Disease or not by the help of features taken as input
    #on the basis of prediction the results are displayed
    if st.button("Predict"):
        if str(prediction3)=="Presence":
            st.warning("You Might Be Affected By Diabetes")
        elif str(prediction3)=="Absence":
            st.success("You Are Safe")





#breast cancer
df = pd.read_csv('processed_data.csv')
X = df.drop('diagnosis', axis = 1)
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

rand_clf = RandomForestClassifier(criterion = 'entropy', max_depth = 10, max_features = 'auto', min_samples_leaf = 2, min_samples_split = 3, n_estimators = 130)
rand_clf.fit(X_train, y_train)



if rad=="Breast_cancer":
    st.header("Know If You Are Affected By Liver")
    st.write("All The Values Should Be In Range Mentioned")

    texture_mean=st.number_input("enter the texture mean from 9.5 to 40")
    smoothness_mean=st.number_input("enter the smoothnes mean from 0.052 to 0.17")
    compactness_mean=st.number_input("enter the compactness mean from 0.019 to .35")
    concave_points_mean=st.number_input("enter the concave mean from 0.0 to 0.21")
    symmetry_mean=st.number_input("enter the symmetry mean from 0.1 to 0.35")
    fractal_dimension_mean=st.number_input("enter the fractal dimension mean from 0.48 to 0.1")
    texture_se=st.number_input("enter the texture se from .35 to 5.0")

    area_se=st.number_input("enter the area se from 7 to 545")
    smoothness_se=st.number_input("enter the smoothness se from 0.0015 to 0.04")

    compactness_se=st.number_input("enter the compactness se from 0.002 to 0.15")
    concavity_se=st.number_input("enter the concavity se  from 0.0 to 4")
    concave_points_se=st.number_input("enter the concave_point se from 9.5 to 40")
    symmetry_se=st.number_input("enter the symmentry se ")
    fractal_dimension_se=st.number_input("enter the fractal dimension se ")

    texture_worst=st.number_input("enter the texture worst")
    area_worst=st.number_input("enter the area worst")
    smoothness_worst=st.number_input("enter the smoothness worst")
    compactness_worst=st.number_input("enter the compactness worst ")
    concavity_worst=st.number_input("enter the concavity worst ")
    concave_points_worst=st.number_input("enter the concave points worst ")
    symmetry_worst=st.number_input("enter the symmetry worst ")
    fractal_dimension_worst=st.number_input("enter the fractal dimension worst ")
    

    
    prediction2=rand_clf.predict([[texture_mean,smoothness_mean,compactness_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,texture_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,texture_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst]])[0]
    
    #prediction part predicts whether the person is affected by Diabetes or not by the help of features taken as input
    #on the basis of prediction the results are displayed
    if st.button("Predict"):
        if prediction2==1:
            st.warning("Manignet")
        elif prediction2==0:
            st.success("Bengin")















#Liver
df4=pd.read_csv('indian_liver_patient.csv')
df4 = df4.drop_duplicates()
df4=df4.dropna(how='any')  

df4 = df4.drop(['Age','Gender'],axis=1)
y=df4.Dataset
x=df4.drop(['Dataset'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2, 
                                                    random_state=0,
                                                    stratify=df4.Dataset)
model4 = SVC()
model4.fit(X_train,y_train)

if rad=='Liver':
    st.header("Know If You Are Affected By Liver")
    st.write("All The Values Should Be In Range Mentioned")

    total_bilirubin=st.number_input("Enter Your total_bilirubin (0.4-75)",min_value=0,max_value=75,step=1)
    direct_bilirubin=st.number_input("Enter direct bilirubin (0-20)",min_value=0,max_value=20,step=1)
    alkaline_phosphotase=st.number_input("Enter alkaline_phosphotase (60-2200)",min_value=60,max_value=2200,step=1)
    almine_aminotransferase=st.number_input("Enter Your Blood Pressure Rate (10-2000)",min_value=10,max_value=2000,step=1)
    aspartate_aminotransferase=st.number_input("aspartate aminotransferase (10-5000)",min_value=10,max_value=5000,step=1)
    total_protein=st.number_input("total protein (2-10)",min_value=2,max_value=10,step=1)
    albumin=st.number_input("albumin",min_value=0,max_value=6,step=1)
    albumin_and_globulin_ratio=st.number_input("Enter ratio ",min_value=0,max_value=3,step=1)
    
    prediction2=model4.predict([[total_bilirubin,direct_bilirubin,alkaline_phosphotase,almine_aminotransferase,aspartate_aminotransferase,total_protein,albumin,albumin_and_globulin_ratio]])[0]
    
    #prediction part predicts whether the person is affected by Diabetes or not by the help of features taken as input
    #on the basis of prediction the results are displayed
    if st.button("Predict"):
        if prediction2==1:
            st.warning("You Might Be Affected By Liver disease")
        elif prediction2==2:
            st.success("You Are Safe")
    


#heading over to the plots section
#plots are displayed for each disease prediction section 
if rad=="Plots":
    #
    type=st.selectbox("Which Plot Do You Want To See?",["Covid-19","Diabetes","Heart Disease"])
    if type=="Covid-19":
        fig=px.scatter(df1,x="Difficulty in breathing",y="Infected with Covid19")
        st.plotly_chart(fig)

    elif type=="Diabetes":
        fig=px.scatter(df2,x="Glucose",y="Outcome")
        st.plotly_chart(fig)
    elif type=="Heart Disease":
        fig=px.scatter(df3,x="BP",y="Heart Disease")
        st.plotly_chart(fig)
