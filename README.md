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



#### 4.2.1 Theroem 1
Intuitively, this theorem upper bounds the error when we have seen enough data. <br>

Let $xt, 1 \leq t \leq N + N_0$ be generated using a mixture of Gaussians  with wi = 1/k,
for all i,  Let $N_0, N \geq  O(1)k^3d^3 \log d $ and $ C \geq \Omega {(k\log k)^{1/4}}$. Then, the algorithm's output satifies the following error bound:

$$ \mathbb{E}(|| \mu^0 - \mu^* ||^2 ) \leq \dfrac{\max_i  ||\mu_i^* ||^2 }{N^{\Omega(1)}} + O(k^3)(\sigma^2\dfrac{d\log N}{N} + \exp (-C^2/8)(c^2+k)\sigma^2) $$

In the following, we will verify theorem 1 experimentally.

```python
def theorem_1(mu, N, sigma, K, C, d):
    return (np.max(LA.norm(mu,axis=0)) / N) + (K**3 * (sigma**2 * d * np.log(N) / N) + np.exp(-C**2 / 8) * (C**2 + K)* sigma**2)
```
    

