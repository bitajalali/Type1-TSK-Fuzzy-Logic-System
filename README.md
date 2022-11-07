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
1 Clustering of educational data:
First, we divide the data set from seconds (1 to 1500) into three training, evaluation and validation sections:
We apply FCM clustering to training data based on input and output values. Because the formation of a cluster means that the outputs
must be close to each other within that cluster, so if the data is clustered based on only the inputs, the outputs may not be similar
in a cluster.Therefore, we have integrated the inputs and labels and enter them as a feature vector into the input of the FCM algorithm.
Finally, we get the U matrix that contains the degree of belonging of each training data to each cluster, as well as their center and variance.
![ن](https://user-images.githubusercontent.com/115353236/200360717-3327c779-ef33-429d-99c1-9c9697703c5a.PNG)

