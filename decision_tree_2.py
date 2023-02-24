#-------------------------------------------------------------------------
# AUTHOR: Alexander Rodriguez 
# FILENAME: decision_tree_2.py 
# SPECIFICATION: This file will be used to generate decision trees from various csv files, with the goal of training, testing, and outputing the performance of three models based on each training set. The final classification performance of each model will be chosen from an average of the accuracy from ten runs. 
# FOR: CS 4210- Assignment #2
# TIME SPENT: 20 min
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0: #skipping the header
                dbTraining.append (row)

    #transforming the original categorical training features to numbers and add to the 4D array X
    for i, row in enumerate(dbTraining):
        X.append(row[:-1])
        for j, col in enumerate(row[:-1]):
            if col == 'Young' or col == 'Myope' or col == 'Yes' or col == 'Normal':
                X[i][j] = 1
            if col == 'Presbyopic' or col == 'Hypermetrope' or col == 'No' or col == 'Reduced':
                X[i][j] = 2
            if col == 'Prepresbyopic':
                X[i][j] = 3
    

    #transforming the original categorical training classes to numbers and adding to the vector Y
    for i, row in enumerate(dbTraining):
        Y.append(row[-1])
        if row[-1] == 'Yes':
            Y[i] = 1
        elif row[-1] == 'No':
            Y[i] = 0

    #looping training and test tasks 10 times here
    avgAccuracy = 0
    
    for i in range (10):
        
        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)

        #read the test data and add this data to dbTest
        dbTest = []
        correctPredictions = 0
        totalPredictions = 0
        
        #reading the training data in a csv file
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0: #skipping the header
                    dbTest.append (row)

        for data in dbTest:
            #transforming the features of the test instances to numbers following the same strategy done during training,
            #and then using the decision tree to make the class prediction.
            for i, col in enumerate(data[:-1]):
                if col == 'Young' or col == 'Myope' or col == 'Yes' or col == 'Normal':
                    data[i] = 1
                if col == 'Presbyopic' or col == 'Hypermetrope' or col == 'No' or col == 'Reduced':
                    data[i] = 2
                if col == 'Prepresbyopic':
                    data[i] = 3
                    
            if data[-1] == 'Yes':
                data[-1] = 1
            elif data[-1] == 'No':
                data[-1] = 0
            
            #Making prediction with model
            class_predicted = clf.predict([data[:-1]])[0]
             
            #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            
            #keeping track of correct and total predictions
            if class_predicted == data[-1]:
                correctPredictions += 1
            totalPredictions += 1

        #adding accuracy for single run to the total average accuracy
        avgAccuracy += correctPredictions / totalPredictions

    #finding the average of this model during the 10 runs (training and test set)
    #diving by 10 to get the average accuracy for our 10 runs
    avgAccuracy = avgAccuracy / 10

    #print the average accuracy of this model during the 10 runs (training and test set).
    print(f'Final accuracy when training on {ds} : {avgAccuracy}')