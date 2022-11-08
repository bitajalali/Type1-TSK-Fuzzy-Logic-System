# Type1-TSK-Fuzzy-Logic-System
In this project, we have designed a type one tsk fuzzy neural network to solve the regression problem.
- In this model, We performed the rule learning step using FCM clustering.
- We performed the step of learning the weights of the last layer based on the pseudo-inverse method.
- The input function of the network is the Mackey Glass functions.

![mac](https://user-images.githubusercontent.com/115353236/200356432-d7d5f0cc-a742-44c0-827e-b99fe4401e49.PNG)


based on the validation error diagram, we obtain the parameters value used in the problem wich are respectively the number of clusters, the criterion Ɣ (to adjust the coverage of the rules obtained in the neural network) and λ (regulator term which is in the objective function in the pseudo-inverse method to obtain the matrix A)

The results of setting the parameters are briefly shown in the following table:

<table>
  <tr>
    <td>Number of clusters</td>
    <td>λ</td>
    <td>Ɣ</td>
  </tr>
  <tr>
    <td>40</td>
    <td>3.1</td>
    <td>0.4</td>
  </tr>
</table>
  
In the following, the details of the steps and results of the parameter setting section are explained
 in detail.


Work steps:
1. Clustering of educational data:
First, we divide the data set from seconds (1 to 1500) into three training, evaluation and validation sections:
We apply FCM clustering to training data based on input and output values. Because the formation of a cluster means that the outputs
must be close to each other within that cluster, so if the data is clustered based on only the inputs, the outputs may not be similar
in a cluster.Therefore, we have integrated the inputs and labels and enter them as a feature vector into the input of the FCM algorithm.
Finally, we get the U matrix that contains the degree of belonging of each training data to each cluster, as well as their center and variance.
![ن](https://user-images.githubusercontent.com/115353236/200360717-3327c779-ef33-429d-99c1-9c9697703c5a.PNG)


Calculation of fuzzy neural network rules, in this section, we first create fuzzy membership functions for the number of clusters, in which we determine the parameter Ɣ, and then we obtain the optimal amount of this parameter according to the validation error.
The parameter Ɣ actually makes the coverage of all the rules proportionally larger or smaller, in other words, this parameter considers the coverage a little smaller than the cluster itself by changing (increasing), usually the coverage is a bit larger than that cluster. We consider a category that can cover the entire space.
So, in this section, by obtaining the Gaussian functions (membership functions) whose parameters are obtained using the FCM clustering results of the training data, according to the following code, the rules obtained from the evaluation and validation data can be obtained.
And then we normalize these rules:

# $e^{\dfrac{Ɣ^{(x-v)^2}}{var^2}}$


```python
u_test=np.zeros((int(c_n),int(len_test)))
for i in range(c_n):
   temp=((test_data[:,0]-mean_var[i,0])**2)/mean_var[i,1]
   u_test[i,:]=np.exp(-gama*temp)

```
So up to this point, the rules have been obtained in the fuzzy neural network.
Now, in this section, we will complete the fuzzy neural network model, which in TSK type 1 is the same as matrix A, 
which we obtain using training data and pseudo-inverse method. In this method, it is obtained by using the formula 
against the coefficients of a in the matrix A:

## $A={(X^T+λI)}^{-1} +X^T.Y^*$

```python
temp=np.matmul(X_train.T,X_train)+(np.identity(c_n*2)*lam)
    term1=np.linalg.inv(temp)
    term2=np.matmul(X_train.T,true_label)
    A=np.matmul(term1,term2)

```
In the mentioned formula, a regulating term called λ is added, which has two advantages:
- It causes the values of A not to be too large in the optimization function.
- It is possible that X. 𝑿𝑻 has a zero eigenvalue, so when it is added to λ, 
the eigenvalues become positive and thus it becomes invertible.
The way to determine the optimal value of λ: for different values of λ, we calculate the matrix A and examine the validation error.

![plot](https://user-images.githubusercontent.com/115353236/200554870-903e911e-8638-47d0-9d20-9ce626a625f3.PNG)

3. Predicting the values of educational data and evaluation and validation:
Now, the model (matrix A) has been optimized and built using the training data values with the pseudo-inverse method. Therefore, data prediction will be done using the formula: Y=X.A.
We obtain the MSE error for the training and evaluation data according to the following figures. Also, by using the validation error, we can determine the optimal coefficient for the parameters of the number of clusters and Ɣ.

![plot](https://user-images.githubusercontent.com/115353236/200556294-94693856-c0cc-43cd-a84d-0d9f118d4627.PNG)
Checking and analyzing the validation chart Ɣ and the number of clusters:
According to the above-mentioned Gaussian membership function formula, as the parameter Ɣ increases or decreases, all the coverage rules become larger or smaller in the same proportion. Based on the clustering of the training data, we have obtained a standard deviation for each cluster. Now, if we want to make this interval a smaller or larger value for all the rules, the value of this parameter can be adjusted, and here we have obtained the optimal value by checking the validation error.
As it can be concluded from the formula of the belonging function, the lower the amount of the super parameter Ɣ is, the coverage of all the rules increases by the same amount, and it can be seen in the graph related to the super parameter Ɣ that for the initial values of 0.1 to 0.3 (which is very low) The coverage of the 
rules has increased to such an extent that it has damaged the accuracy of the model and the validation error has increased significantly, but the most optimal amount is 0.4. The values will be less and the model will cover less points between the points and as a result the error will increase.
In this way, we will train the model by using the optimal values obtained by the validation error, and we will ultimately reduce the training and evaluation error 
to a minimum: 
Fitted graph of the Mackey_Glass function:

![plot](https://user-images.githubusercontent.com/115353236/200562044-3d1c0ebc-8a75-4a11-aec8-f24614b578d4.PNG)

As it can be seen from the regression results, the fuzzy neural network with the help of logic and fuzzy system has been able to achieve the results with a limited number of training data, unlike deep neural networks that require a large number of training data to train the model. and achieve good accuracy.

- A suggestion to improve the performance of the system for this regression problem:
In general, using the clustering method in the fuzzy neural network, where FCM clustering is used here, is more appropriate in classification problems than regression.
In the regression problem, it is better that the centers of the rules (the centers of the belonging functions) are the extreme points of the function. Because clustering does not guarantee to determine these points as cluster centers. So, my suggestion to achieve more accurate results is to find the extreme points of the function in the training data section and determine them as the centers of the rules.
    

