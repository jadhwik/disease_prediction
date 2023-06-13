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




def description(out):
    filename = r'C:\Users\91984\Desktop\project\datasets\symptom_Description.csv'
    df = pd.read_csv(filename)
    le = LabelEncoder()
    df['Disease'] = le.fit_transform(df['Disease'])
    details = dict(zip(df['Disease'], df['Description']))
    result = out[0]
    if result in details:
        # st.write('Printing disease description=========================================')
        st.info(details[result])

def precaution(out, le):
    filename = r'C:\Users\91984\Desktop\project\datasets\precaution.csv'
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
    # st.write('prediction using Naive Bayes')
    # st.write(out)
    # st.write(le.inverse_transform(out))
    st.success('Predicted disease: ' + le.inverse_transform(out)[0])
    description(out)
    precaution(out, le)

     # #implementing decision tree
def decision_tree(X_train,X_test,y_train,y_test,a,le):
     from sklearn import tree

     tr=tree.DecisionTreeClassifier(criterion='entropy',max_depth=20)
     tr.fit(X_train,y_train)
     sc=tr.score(X_test,y_test)
     y_pred=tr.predict(X_test)
     print(sc)
    #  print(classification_report(y_test,y_pred))
    #  print(accuracy_score(y_test,y_pred))
     out=tr.predict(a)
     print('prediction decision tree')
     print(out)
     st.success( le.inverse_transform(out)[0])
     description(out)
     precaution(out,le)



def random_forest(X_train,X_test,y_train,y_test,a,le):
     from sklearn.ensemble import RandomForestClassifier
     rc=RandomForestClassifier(n_estimators=2)
     rc.fit(X_train,y_train)
     print('prediction using random forest')
     print(rc.score(X_test,y_test))
     print(rc.predict(X_test))
     out=rc.predict(a)
     print(out)
     st.success(le.inverse_transform(out)[0])
     description(out)
     precaution(out,le)



def gradientboost(X_train,X_test,y_train,y_test,a,le):
     from sklearn.ensemble import GradientBoostingClassifier
     gbc=GradientBoostingClassifier(learning_rate=0.1)
     gbc.fit(X_train,y_train)
     print('prediction using gradient boosting')
     print(gbc.score(X_test,y_test))
     print(pd.DataFrame(gbc.predict(X_test)))
     out=gbc.predict(a)
     print(out)
     st.success(le.inverse_transform(out)[0])
     description(out)
     precaution(out,le)

def main():
    
    
    # Load the dataset
    filepath = r"C:\Users\91984\Desktop\project\training.csv"
    df = pd.read_csv(filepath)
    le = LabelEncoder()
    df['prognosis'] = le.fit_transform(df['prognosis'])

    # Retrieve the feature data from the dataset
    X = df.iloc[:, :-2].values
    y = df['prognosis'].values

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Read the symptom data for prediction
    filepath = r"C:\Users\91984\Desktop\project\testing.csv"
    df1 = pd.read_csv(filepath)
    df1['prognosis'] = le.transform(df1['prognosis'])
    a = df1.iloc[:, :-2]

   
    st.header('DISEASE PREDICTION')
    
    
    symptom_names = df.columns[:-2]
    
    
    filtered_symptoms = [symptom for symptom in symptom_names]
    

    selected_symptoms = st.multiselect('Symptoms', filtered_symptoms, [], key='symptoms')
    

    # # selected_symptoms = st.multiselect('Symptoms', symptom_names, [], key='symptoms')[:5]
    

    # while len(selected_symptoms) < 5:
    #     selected_symptoms.append('')

    a = [1 if symptom in selected_symptoms else 0 for symptom in symptom_names]
    a = [a]  # Convert to 2D array

    st.sidebar.markdown('RESULT')
    if st.sidebar.button('PREDICTION1', key='button1',help='Button1'):
        naive_bayes(X_train, X_test, y_train, y_test, le, a)
        
        pass

    if st.sidebar.button('PREDICTION2', key='button2',help='Button2'):
        decision_tree(X_train,X_test,y_train,y_test,a,le)
       
        pass

    if st.sidebar.button('PREDICTION3', key='button3',help='Button3'):
        random_forest(X_train,X_test,y_train,y_test,a,le)
        pass

    if st.sidebar.button('PREDICTION4', key='button4',help='Button4'):
        gradientboost(X_train,X_test,y_train,y_test,a,le)    
        pass
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
    
