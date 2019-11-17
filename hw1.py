import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn
import copy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
'''
Question 2
In this question, you will learn how to perform polynomial regression with linear
regression tools using python.
Given train data (𝑥1 = 0, 𝑦1 = −1.25), (𝑥2 = 0.5, 𝑦2 = −0.6), (𝑥3 = 2, 𝑦3 = −4.85)
and test data (𝑥4 = −1, 𝑦4 = −5.2), (𝑥5 = 1, 𝑦5 = −0.9), (𝑥6 = 3, 𝑦6 = −13):

a. Define a 3 × 1 two-dimensional matrix called X_train in which each line is an
observation from the training data, and define a (one-dimensional) vector called
y_train which contains the responses of these observations (in the same order).
Repeat this process with the test data (X_test, y_test).
'''
X_train = np.array([[0],[0.5],[2]])
y_train = np.array([-1.25,-0.6,-4.85])
X_test = np.array([[-1],[1],[3]])
y_test = np.array([-5.2,-0.9,-13])

'''
b. Calculate the regular LS estimators 𝑤̂0, 𝑤̂1 using only the training data with
sklearn built-in functions. What are the predicted values for X_test?
'''
regress = linear_model.LinearRegression()
regress.fit(X_train, y_train)
y_pred = regress.predict(X_test)
print(y_pred)
print(regress.score(X_test, y_test))
print('Coefficients: \n', regress.coef_)
print('Intercept: \n', regress.intercept_)

'''
c. What is the MSE of the regression on the train data? What is the MSE of the
regression on the test data? What can you conclude from these values?
'''
from sklearn.metrics import mean_squared_error
print("MSE on Train is: ",mean_squared_error(y_train,y_pred))
print("MSE on Test is: ",mean_squared_error(y_test,y_pred))
print("Not a good prediction")
'''
d. Write a function which receives a np-array of explanatory variables X and a
np-array of responses Y and returns the least squares estimator using the closed form expression we saw in class. 
There is no need to check that the input is valid.
(don’t forget to add ones!)
'''

def estimate_coef(x, y):
    x = np.column_stack((x, np.ones(len(x))))
    ones = np.ones(y.shape)
    np.append(y, ones, axis=0)
    coeffs = inv(x.transpose().dot(x)).dot(x.transpose()).dot(y)
    intercept = coeffs[-1]
    new_coeffs = coeffs[:-1]
    print(f"coeffs:{new_coeffs}")
    print(f"intercept:{intercept}")
    return coeffs

'''
e. Plot the regression line in a dashed (--) black line. Scatter (with plt.scatter()) the
points in the train data with marker='*' and scatter the points in the test data
with marker='o'. You should use legend (for train and test data) and label the
axes. The range in the x axis should be np.arange(-3, 5).
Does it look like the regression fit the data?
'''
plt.scatter(X_train, y_train,color="green", marker='*', s=10, label='Train data')
plt.scatter(X_test, y_test,color="blue", marker='o', s=10, label='Test data')
plt.plot(X_test, y_pred, color='black', linestyle='--')
plt.xlim(-3, 5)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
'''
f. We will now try to perform a 2nd degree polynomial regression, 
meaning we will assume now that 𝑦𝑖 ≈ 𝛾0 + 𝛾1 ∙ 𝑥𝑖 + 𝛾2 ∙ 𝑥𝑖2
1. If we would mark 𝑧𝑖 = (𝑥𝑖, 𝑥𝑖2)𝑇, how can we write 𝑦𝑖 in a linear form?
2. Define a 3 × 2 matrix called Z_train in which the first column corresponds to
𝑥𝑖 and the second column corresponds to 𝑥𝑖2
for each 𝑥𝑖in the train data (in the same order as in section a.). Repeat this process with the test data (Z_test).
3. Use the function you wrote in section d. to calculate the LS estimators 𝛾̂ =(𝛾̂0, 𝛾̂1, 𝛾̂2)
𝑇 using only the training data (Z_train and y_train). What is the
MSE of the regression on the train data? What is the MSE on the test data?
4. Plot the corresponding 2nd degree polynomial function in a red dashed line,
alongside the original regression line in a black dashed line. Scatter the
points in the training data (with marker='*'), as well as the points in the
testing data (with marker='o'). You should use legend for both data type
(train/test) and regression type (linear/polynomial) and label the axes. The
range in the x axis should be np.arange(-3, 5). Name the plot 'Polynomial
Regression vs. Linear Regression'.
Which regression seems to perform better?
g. Which assumption did not hold, thus making the linear regression to fail?
'''
Z_train = np.array([[0, 0], [0.5, 0.25], [2, 4]])
Z_test = np.array([[-1, 1], [1, 1], [3, 9]])

new_coeffs = estimate_coef(Z_train, y_train)

Z_pred = new_coeffs[0] * Z_test[:, 0] + new_coeffs[1] * Z_test[:, 1] + new_coeffs[-1]

print("MSE of test: %.2f" % np.average((y_test - Z_pred) ** 2))
print("MSE of train: %.2f" % np.average((y_train - Z_pred) ** 2))

plt.scatter(X_train, y_train, color="green", marker='*', s=10, label='Train data')
plt.scatter(X_test, y_test, color="blue", marker='o', s=10, label='Test data')
plt.plot(X_test, y_pred, color='black', label='linear')
x = np.linspace(-3,5,100)
fx = []
for i in range(len(x)):
    fx.append(new_coeffs[1] * x[i] ** 2+new_coeffs[0] * x[i]+new_coeffs[-1])
plt.plot(x, fx, color='red', label='polynomial')
plt.xlim(-3, 5)
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Regression vs. Linear Regression')
plt.show()

"""
Question 3:
The course website contains a file called "parkinsons_updrs_data.csv" in which there are 5875 records.
 This file contains the attributes of speech recordings of 42 individuals with Parkinson's disease. 
 In this question, we will try to predict the medical status of people with Parkinson's disease using their speech data.
  A file called "parkinsons_updrs.names.txt" with more details on the data is also in the website.
a. Load the data using the Pandas library.
b. How many males are in the data? How many females? Show a bar plot.
c. Visualize the distribution of the ages with a histogram.
d. How does the mean of the variable "motor_UPDRS" varies between different sexes?
   Visualize it using a bar plot (use groupby).
e. Choose 6 explanatory variables and display a scatter plot of them with the response variable motor_UPDRS 
    (use the function scatter_matrix()).
f. Use sklearn's built-in functions to calculate the LS estimator, 
    using only the 6 explanatory variables you chose in the previous section.
g. Use the function from question 2 to calculate the least squares estimator, 
    using only the 6 explanatory variables you chose in the section 
e. Did you get the same answer as in the previous section? 
    (hint: the answer should be yes. Otherwise, you have a problem)
"""

objects= ('female','male')
y_pos=np.arange(len(objects))
df=pd.read_csv('parkinsons_updrs_data.csv')
df.groupby('sex')['sex'].aggregate(lambda x: x.count()).plot(kind='bar')
plt.xticks(y_pos,objects)

#c
df.groupby('age')['age'].count().plot(kind='bar')

df.groupby('sex')['motor_UPDRS'].aggregate(lambda x: x.sum() / len(x)).plot(kind='bar')
plt.xticks(y_pos,objects)
#האם קריטי טווח המספרים של ציר Y

#e
pd.plotting.scatter_matrix(df.loc[:,
                                  ['motor_UPDRS','Jitter.Per','Jitter.RAP','Jitter.PPQ5','Shimmer.APQ3','Shimmer.dB','RPDE']])
plt.show()

#f
data=df.loc[:,['Jitter.Per','Jitter.RAP','Jitter.PPQ5','Shimmer.APQ3','Shimmer.dB','RPDE']]
reg=LinearRegression().fit(data,df.loc[:,['motor_UPDRS']])
w0=reg.intercept_[0]
w1=reg.coef_[0,0]
print(f"w0={w0}, w1={w1}")

#f
def Least_Squares(x, y):
    return inv(x.transpose().dot(x)).dot(x.transpose().dot(y))

df1 = df
df1['one'] = 1
array = Least_Squares(df[['Jitter.Per','Jitter.RAP','Jitter.PPQ5','Shimmer.APQ3','Shimmer.dB','RPDE', 'one']].values, df.motor_UPDRS)

print(array)

"""
Question 4:
In this question you will implement the Perceptron algorithm:
1. 𝑤(1)=(0,…,0)𝑇.
2. For 𝑡=1 to T:
2.1 If there exists 𝑖 such that 𝑦𝑖∙𝑤𝑇𝑥𝑖≤0:
2.1.1 Set 𝑤(𝑡+1)=𝑤(𝑡)+𝑦𝑖𝑥𝑖.
2.1.2 Continue.
2.2 Else return 𝑤(𝑡).
Write a function called Perceptron which receives the following parameters:
    X – a two-dimensional matrix (np-array) in ℝ𝑛×𝑑 which contains the observations.
    y – a vector (np-array) in ℝ𝑛 which contains the responses.
    The function should return 𝑤 after the convergence of the Perceptron algorithm.
Notes:
1. This function should be generic (meaning it can receive data in any dimension).
2. You may assume that the data is linearly separable, and you are not required to check that the input is valid.
3. Do not forget to add a constant 1 to all observations – 
    the function should produce a 
    (d+1)-dimensional vector in which 𝑤0 is the intercept and 𝑤𝑖 is the coefficient of 𝑥𝑖 for each 𝑖≥1.
4. You will use this function in the next exercises. For validating the quality of your function, 
    you can use synthetic data from here: http://scikit-learn.org/stable/datasets/index.html.
"""
def perceptron(data):
    lab_data = {'x': data.data, 'y': data.target}
    w = np.zeros(len(lab_data['x'][0]))
    eta = 1
    epochs = 20
    for t in range(epochs):
        for i, x in enumerate(lab_data['x']):
            if(np.dot(lab_data['x'][i], w)*lab_data['y'][i]) <= 0:
                w += eta*lab_data['x'][i]*lab_data['y'][i]

    return w

"""
Querstion 5:
In this question you will use the sklearn's Iris dataset from which contains 150 observations of 3 Iris species. 
    You will use Logistic Regression classifier in order to generate a multi-class classifier of type one-versus-rest.
a. Load the data using the following code:
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
X is a np-array with 150 rows and 4 columns: Sepal Length, Sepal Width, Petal Length and Petal Width.
y is a np-array with 150 rows and 1 column which contains 0, 1, or 2
    for each Iris species: Setosa, Versicolour and Virginicacv (respectively).
b. Split the data to train and test data. Use the train_test_split function with random_state=1000.
c. A Logistic Regression classifier is a binary classifier, which also provides us a confidence level – 
    the probability of belonging to each class of the two. In the following sections, 
    you will build a one-versus-rest classifier using the following scheme:
o Create a new vector from the response vector (y_train) which contains 1 for the Setosa and -1 for the other species. 
    Do the same for y_test.
o Build a Logistic Regression classifier for the species Setosa (using the training data). 
    Use the new response vector you have created.
d. Repeat section c for the other two species.
e. Create a function which receives all the above classifiers and a collection of observations as np-array 
    and returns a one-versus-rest classification vector, 
    in which each observation is classified to the class for which it has the maximal probability to belong to.
f. Use the function you wrote to perform a one-versus-rest classification for the test data. 
    Use the output of this function to create and plot a confusion matrix. 
    In order to calculate it, use the following code:
from sklearn.metrics import confusion_matrix
my_confusion_matrix = confusion_matrix(y_test, y_pred)
You are to display two different plots – 
one for the unnormalized (original) confusion matrix and one for the normalized (as seen in tutorial 3) matrix.
In order to plot it, use the following code:
import seaborn as sn
sn.heatmap(my_confusion_matrix, annot=True, cmap="tab20b")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('…') # change the title according to the type of the confusion matrix
plt.show()
g. Choose one observation which is misclassified,
    and explain why you think the one-versus-rest classifier did not classify it right.
"""
#a

iris = datasets.load_iris()
X = iris.data
y = iris.target
#b

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1000)
#print(y_test)

#c
y_train1 = copy.deepcopy(y_train)
#print(y_train1)
index = 0

for i in y_train1:
    if y_train1[index] == 0:
        y_train1[index] = 1
    else:
        y_train1[index] = -1
    index += 1
#print(y_train1)

y_test1 = copy.deepcopy(y_test)
index = 0

for i in y_test1:
    if y_test1[index] == 0:
        y_test1[index] = 1
    else:
        y_test1[index] = -1
    index += 1

lr1 = LogisticRegression(solver='lbfgs').fit(X_train, y_train1.ravel())
test1 = lr1.predict_proba(X_test)
res1 = np.zeros(shape=(len(X_test)))
for i in range(len(test1)):

    if test1[i][1] >= 0.5:
        res1[i] = 1
    else:
        res1[i] = -1

np.set_printoptions(formatter={'float_kind': '{:f}'.format})

#d

y_train2 = copy.deepcopy(y_train)
index = 0
for i in y_train2:

    if y_train2[index] == 1:
        y_train2[index] = 1
    else:
        y_train2[index] = -1
    index += 1

y_test2 = copy.deepcopy(y_test)

index = 0
for i in y_test2:
    if y_test2[index] == 1:
        y_test2[index] = 1
    else:
        y_test2[index] = -1
    index += 1

lr2 = LogisticRegression(solver='lbfgs').fit(X_train, y_train2.ravel())
test2 = lr2.predict_proba(X_test)
res2 = np.zeros(shape=(len(X_test),))

for i in range(len(test2)):
    if test2[i][1] >= 0.5:
        res2[i] = 1
    else:
        res2[i] = -1

y_train3 = copy.deepcopy(y_train)
index = 0

for i in y_train3:
    if y_train3[index] == 2:
        y_train3[index] = 1
    else:
        y_train3[index] = -1
    index += 1

y_test3 = copy.deepcopy(y_test)
index = 0

for i in y_test3:
    if y_test3[index] == 2:
        y_test3[index] = 1
    else:
        y_test3[index] = -1
    index += 1

lr3 = LogisticRegression(solver='lbfgs').fit(X_train, y_train3.ravel())
test3 = lr3.predict_proba(X_test)
res3 = np.zeros(shape=(len(X_test),))

for i in range(len(test3)):
    if test3[i][1] >= 0.5:
        res3[i] = 1
    else:
        res3[i] = -1


#e

def one_versus_rest(X_test, lr1, lr2, lr3):
    test1 = lr1.predict_proba(X_test)
    test2 = lr2.predict_proba(X_test)
    test3 = lr3.predict_proba(X_test)

    arrpred = np.zeros(shape=(len(X_test),))
    resulte = np.zeros(shape=(len(X_test),))
    for i in range(len(test1)):
        arrpred[i] = max(test1[i][1], test2[i][1], test3[i][1])
        if arrpred[i] == test1[i][1]:
            resulte[i] = 0
        elif arrpred[i] == test2[i][1]:
            resulte[i] = 1
        elif arrpred[i] == test3[i][1]:
            resulte[i] = 2

    return resulte



#f

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn

my_confusion_matrix = confusion_matrix(y_test, one_versus_rest(X_test, lr1, lr2, lr3))
print(my_confusion_matrix)

sn.heatmap(my_confusion_matrix, annot=True, cmap="tab20b")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Unnormalized confusion matrix')
plt.show()

my_confusion_matrix = my_confusion_matrix.astype('float') / my_confusion_matrix.sum(axis=1)[:, np.newaxis]
print(my_confusion_matrix)

sn.heatmap(my_confusion_matrix, annot=True, cmap="tab20b")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Normalized confusion matrix')
plt.show()

#g

