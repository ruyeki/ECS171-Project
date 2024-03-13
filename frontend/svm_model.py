from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
import pickle


#This chunk of code is just copy and paste from the ipynb file, just preprocessing the data all over again
data = pd.read_csv('data.csv')
data = data.dropna()
corr = data.corr()
vars = []

for i in range(len(corr["Bankrupt?"])):
    if abs(corr["Bankrupt?"][i]) > 0.15:
        vars.append(corr["Bankrupt?"].index[i])

data = data[vars]

missing_values = data.isnull().sum()
data.fillna(data.median(), inplace=True)
data = data.drop_duplicates()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.drop(columns=['Bankrupt?']))
scaled_df = pd.DataFrame(scaled_features, columns=data.columns[1:])

X = scaled_df
y = data['Bankrupt?']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)



#This is the final model that we came up with that will be doing the predictions based off user input
final_svc = SVC(C=0.3, class_weight='balanced', gamma=0.02, kernel='rbf')
final_svc.fit(X_train, y_train)

# Save the trained model to a file for use in streamlit
with open('svm_model.pkl', 'wb') as model_file:
    pickle.dump(final_svc, model_file)

