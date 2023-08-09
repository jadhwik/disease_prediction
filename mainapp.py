import streamlit as st
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from streamlit import components
from streamlit_lottie import st_lottie
import json
import requests
import base64
from sklearn.metrics import accuracy_score
import numpy as np



def description(out):
    filename = 'symptom_Description.csv'
    df = pd.read_csv(filename)
    le = LabelEncoder()
    df['Disease'] = le.fit_transform(df['Disease'])
    details = dict(zip(df['Disease'], df['Description']))
    result = out[0]
    if result in details:
        # st.write('Printing disease description=========================================')
        st.info(details[result])

def precaution(out, le):
    filename ='precaution.csv'
    df = pd.read_csv(filename)
    df['DISEASE'] = le.fit_transform(df['DISEASE'])
    details = dict(zip(df['DISEASE'], df['PRECAUTION']))
    if out[0] in details:
        # st.write('Printing disease precaution=========================================')
        st.info('Precautions you must take:'+details[out[0]])

def naive_bayes(X_train, X_test, y_train, y_test, le, a):
    model = GaussianNB()
    model.fit(X_train, y_train)
    out = model.predict(a)
    print('prediction using naive bayes----------------------')
    # st.write('prediction using Naive Bayes')
    # st.write(out)
    # st.write(le.inverse_transform(out))
    st.success('Predicted disease: ' + le.inverse_transform(out)[0])
    description(out)
    precaution(out, le)
    accuracy = accuracy_score(y_test, model.predict(X_test))*100
    st.write(" Accuracy:", accuracy)

    return {"disease": le.inverse_transform(out)[0], "accuracy": accuracy}
    

     # #implementing decision tree
def decision_tree(X_train,X_test,y_train,y_test,a,le):
     from sklearn import tree

     tr=tree.DecisionTreeClassifier(criterion='entropy',max_depth=20)
     tr.fit(X_train,y_train)
    #  sc=tr.score(X_test,y_test)
    #  y_pred=tr.predict(X_test)
    
    #  print(classification_report(y_test,y_pred))
    #  print(accuracy_score(y_test,y_pred))
     out=tr.predict(a)
     print('prediction decision tree')
     print(out)
     st.success( le.inverse_transform(out)[0])
     description(out)
     precaution(out,le)
     accuracy = accuracy_score(y_test, tr.predict(X_test))*100
     st.write(" Accuracy:", accuracy)
     return{"disease": le.inverse_transform(out)[0], "accuracy": accuracy}
     

def random_forest(X_train,X_test,y_train,y_test,a,le):
     from sklearn.ensemble import RandomForestClassifier
     rc=RandomForestClassifier(n_estimators=2)
     rc.fit(X_train,y_train)
     print('prediction using random forest')
    #  print(rc.score(X_test,y_test))
    #  print(rc.predict(X_test))
     out=rc.predict(a)
     print(out)
     st.success(le.inverse_transform(out)[0])
     description(out)
     precaution(out,le)
     accuracy = accuracy_score(y_test, rc.predict(X_test))*100
     st.write(" Accuracy:", accuracy)
     return {"disease": le.inverse_transform(out)[0], "accuracy": accuracy}





def main():
    
    
    # Load the dataset
    filepath = "Training.csv"
    df = pd.read_csv(filepath)
    le = LabelEncoder()
    df['prognosis'] = le.fit_transform(df['prognosis'])

    # Retrieve the feature data from the dataset
    X = df.iloc[:, :-2].values
    y = df['prognosis'].values

    # Split the dataset into train and test sets
    from sklearn.model_selection import StratifiedKFold
    fold=StratifiedKFold(n_splits=3,shuffle=True,random_state=42)

    for test_index,train_index in fold.split(X,y):
    
        X_test,X_train,y_test,y_train=X[test_index],X[train_index],y[test_index],y[train_index]


    # Standardize the data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Read the symptom data for prediction
    filepath ="Testing.csv"
    df1 = pd.read_csv(filepath)
    df1['prognosis'] = le.transform(df1['prognosis'])
    a = df1.iloc[:, :-2]

   
    st.header('DISEASE PREDICTION')
    
    
    symptom_names = df.columns[:-2]
    
    
    filtered_symptoms = [symptom for symptom in symptom_names]
    

    selected_symptoms = st.multiselect('Symptoms', filtered_symptoms, [], key='symptoms')
    

    a = [1 if symptom in selected_symptoms else 0 for symptom in symptom_names]
    a = [a]  # Convert to 2D array

    st.sidebar.markdown('RESULT')
    
    

    predicted_results=[]
    def b1():
        return naive_bayes(X_train, X_test, y_train, y_test, le, a)
    def b2():
        return decision_tree(X_train, X_test, y_train, y_test, a, le)
        
    def b3():    
        return random_forest(X_train, X_test, y_train, y_test, a, le)
   
    


    
    def results(predicted_results):
        predicted_results.append(b1())
        predicted_results.append(b2())
        predicted_results.append(b3())
        
    
        
    main_result=[]
    if st.sidebar.button('PREDICTION1', key='button1'):
        b1()
    if st.sidebar.button('PREDICTION2', key='button2'):
        b2()
    if st.sidebar.button('PREDICTION3', key='button3'):
        temp=[]
        for func in [b1, b2, b3]:
            result = func()
            temp.append(result)
        main_result.extend(temp)
        print(main_result)
    

        
        if len(main_result) ==3:
            # Sort the predicted results by accuracy in descending order
            main_result = sorted(main_result, key=lambda x: x["accuracy"], reverse=True)

            # Get the best predicted result
            best_result = main_result[0]
            best_disease = best_result["disease"]
            best_accuracy = best_result["accuracy"]

            # Print the best predicted disease and accuracy
            st.sidebar.markdown('THE ACCURATE DISEASE MAY BE:')
            st.sidebar.success(f"Best Predicted Disease: {best_disease}")
            st.sidebar.success(f"Accuracy: {best_accuracy}")
    

    

    
    
    
    animation_url = "https://assets7.lottiefiles.com/packages/lf20_2ZKqKUm2Jm.json"
    set_background_animation(animation_url)
   

    

def set_background_animation(animation_url):
    response = requests.get(animation_url)
    if response.status_code == 200:
        animation_data = response.json()
        st_lottie(animation_data, speed=1, height=400, key="lottie-animation")
    else:
        st.error("Failed to load animation.")

if __name__ == '__main__':
    main()
    
