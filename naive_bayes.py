#-------------------------------------------------------------------------
# AUTHOR: Alexander Rodriguez
# FILENAME: naive_bayes.py
# SPECIFICATION: Program reads weather_training.csv and outputs classification of
#    each test instance from the file weather_test.csv if confidence is >= 0.75
# FOR: CS 4210- Assignment #2
# TIME SPENT: 20 min
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

db = []

#reading the training data in a csv file
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            db.append (row)

#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
X = []

for i, row in enumerate(db):
    X.append(row[1:-1])
    for j, col in enumerate(X[i]):
        if X[i][j] == 'Sunny' or X[i][j] == 'Hot' or X[i][j] == 'High' or X[i][j] == 'Weak':
            X[i][j] = 0
        elif X[i][j] == 'Overcast' or X[i][j] == 'Mild' or X[i][j] == 'Normal' or X[i][j] == 'Strong':
            X[i][j] = 1
        elif X[i][j] == 'Rain' or X[i][j] == 'Cool':
            X[i][j] = 2

#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
Y = []
for i, row in enumerate(db):
    if row[-1] == 'Yes':
        Y.append(1)
    else:
        Y.append(0)

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

db_test = []
#reading the test data in a csv file
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            db_test.append (row)

X_test = []

for i, row in enumerate(db_test):
    X_test.append(row[1:-1])
    for j, col in enumerate(X_test[i]):
        if X_test[i][j] == 'Sunny' or X_test[i][j] == 'Hot' or X_test[i][j] == 'High' or X_test[i][j] == 'Weak':
            X_test[i][j] = 0
        elif X_test[i][j] == 'Overcast' or X_test[i][j] == 'Mild' or X_test[i][j] == 'Normal' or X_test[i][j] == 'Strong':
            X_test[i][j] = 1
        elif X_test[i][j] == 'Rain' or X_test[i][j] == 'Cool':
            X_test[i][j] = 2
            
#printing the header os the solution
h = ['Day:','Outlook:','Temp:','Humidity:','Wind:','PlayTennis:','Confidence:']
print(f'{h[0]:<7}{h[1]:<10}{h[2]:<8}{h[3]:<12}{h[4]:<8}{h[5]:<14}{h[6]:<10}')

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
for i, row in enumerate(X_test):
    class_probs = clf.predict_proba([row])[0]
    pred_class = None
    str_class = None
    
    if class_probs[0] > class_probs[1]:
        pred_class = 0
        str_class = 'No'
    else:
        pred_class = 1
        str_class = 'Yes'
        
    if class_probs[pred_class] >= 0.75:
        print(f'{db_test[i][0]:<7}{db_test[i][1]:<10}{db_test[i][2]:<8}{db_test[i][3]:<12}{db_test[i][4]:<8}{str_class:<14}{class_probs[pred_class]:<10.2}')

