import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import random
import math
from scipy.spatial.distance import cdist
random.seed(1000)
##########################################################################
##########################################################################


def mackey(n_iters):
    am = 0.2
    bm = 1
    cm = 0.9
    dm = 30
    hm = 10
    x0 = 0.23
    x = np.zeros((n_iters,))
    x[0:dm] = x0 * np.ones((dm,))
    for i in range(dm, n_iters - 1):
        a = x[i]
        b = x[i - dm]
        y = ((am * b) / (bm + b ** hm)) + cm * a
        x[i + 1] = y
    return x

D = 4  # number of regressors
T = 1  # delay
N = 1500 # Number of points to generate
mg_series = mackey(N)  # Use last 1500 points



x=np.arange(0,N)

total_data = np.vstack((x,mg_series)).T
plt.scatter(total_data[:,0],total_data[:,1])
plt.title('MACKEY GLASS TIME SERIE')
plt.show()
#################SHUFFELING#########################################

random.shuffle(x)
total_data=total_data[x]
###################################################################
len_train=0.6*len(x)
len_test=0.3*len(x)
len_val=0.1*len(x)
####################################################################
train_data=total_data[:int(len_train)]
test_data=total_data[int(len_train):-int(len_val)]
val_data=total_data[-int(len_val):]

colors = ['Navy','DarkBlue','B','MediumBlue','Blue','MidnightBlue','RoyalBlue','SteelBlue','DodgerBlue','DeepSkyBlue','CornflowerBlue','SkyBlue','LightSkyBlue','LightSteelBlue',	'LightBlue','PowderBlue','deeppink','deepskyblue','MediumVioletRed','DeepPink','PaleVioletRed','HotPink','LightPink','Pink','DarkRed','Red','Firebrick','Crimson','IndianRed','LightCoral','Salmon','DarkSalmon','LightSalmon','orange', 'g', 'r', 'c', 'm', 'y', 'k','Brown', 'ForestGreen']
true_label=train_data[:,1]

def mse_error(gama,c_n,lam,bias=1):


#####MAKING U MATRIX FOR TRAIN DATA#########

        cntr,u_train, u0, d, jm, p, fpc=fuzz.cluster.cmeans(train_data.T, c_n, 2, error=0.005, maxiter=1000, init=None, seed=None)
        cluster_membership = np.argmax(u_train, axis=0)
        '''
        for j in range(c_n):
            plt.scatter(train_data[cluster_membership==j,0],train_data[cluster_membership==j,1],c=colors[j])
        plt.title('MACKEY GLASS TIME SERIE CLUSTERED TRAIN DATA WITH FCM ')
        plt.show()
        '''

##########COMPUTING MEAN AND VARIANCES OF CLUSTERS OF TRAIN DATA###########

        means=[]
        vars=[]
        for  i in range(u_train.shape[0]):
            u_train[i,:]=(u_train[i,:]**2)/np.sum((u_train[i, :]**2))
            mean = (np.matmul(train_data[:,0], (u_train[i, :])))
            var=np.matmul(u_train[i,:],((train_data[:,0] - mean)**2))
            means.append(mean)
            vars.append(var)
        mean_var=np.vstack((means,vars)).T
################COMUTING U MATRIX OF TEST DATA##COMPUTING GUSSIAN FUNCTION###################


        u_test=np.zeros((int(c_n),int(len_test)))
        for i in range(c_n):

            temp=((test_data[:,0]-mean_var[i,0])**2)/mean_var[i,1]
            u_test[i,:]=np.exp(-gama*temp)

################COMUTING U MATRIX OF VALIDATION DATA## COMPUTING GUSSIAN FUNCTION#####################

        u_val=np.zeros((int(c_n),int(len_val)))
        for i in range(c_n):

            temp=((val_data[:,0]-mean_var[i,0])**2)/mean_var[i,1]
            u_val[i,:]=np.exp(-gama*temp)
###############COMUTING X MATRIX OF TRAIN DATA###########################


        X_train=np.zeros((u_train.shape[1],u_train.shape[0]*2))

        for i in range(u_train.shape[1]):
            g=u_train[:, i]/np.mean(u_train[:,i])
            g_array =g.repeat(2)
            x_array = np.tile([bias,train_data[i,0]], c_n)
            X_train[i,:]=np.multiply(x_array,g_array).reshape(1,-1)
##################COMPUTING X MATRIX OF TEST DATA########################


        X_test=np.zeros((u_test.shape[1],u_test.shape[0]*2))

        for i in range(u_test.shape[1]):
            g=u_test[:, i]/np.mean(u_test[:,i])
            g_array =g.repeat(2)
            x_array = np.tile([bias, test_data[i,0]], c_n)
            X_test[i,:]=np.multiply(x_array,g_array).reshape(1,-1)

##################COMPUTING X MATRIX OF VALIDATION DATA########################

        X_val=np.zeros((u_val.shape[1],u_val.shape[0]*2))

        for i in range(u_val.shape[1]):
            g=u_val[:, i]/np.mean(u_val[:,i])
            g_array =g.repeat(2)
            x_array = np.tile([bias, val_data[i,0]], c_n)
            X_val[i,:]=np.multiply(x_array,g_array).reshape(1,-1)
################### MAKING MODEL ###################
        temp=np.matmul(X_train.T,X_train)+(np.identity(c_n*2)*lam)
        term1=np.linalg.inv(temp)

        term2=np.matmul(X_train.T,true_label)
        A=np.matmul(term1,term2)
###########################################

############## PREDICTING TRAIN AND TEST DATA ###########

        train_predicted_label=np.matmul(X_train,A)
        test_predicted_label=np.matmul(X_test,A)
        val_predicted_label=np.matmul(X_val,A)
        train_predicted=np.vstack((train_data[:,0],train_predicted_label)).T
        test_predicted=np.vstack((test_data[:,0],test_predicted_label)).T
        val_predicted=np.vstack((val_data[:,0],val_predicted_label)).T
        predicted=np.concatenate((train_predicted,test_predicted,val_predicted))
        train_error = train_data[:, 1] - train_predicted_label
        train_mse_error = (np.matmul(train_error, train_error)) / len(train_error)
        test_error = test_data[:, 1] - test_predicted_label
        test_mse_error = (np.matmul(test_error, test_error)) / len(test_error)
        val_error = val_data[:, 1] - val_predicted_label
        val_mse_error = (np.matmul(val_error, val_error)) / len(val_error)
        # return predicted,train_error,test_error,val_error
        return [val_mse_error,predicted,train_mse_error,train_mse_error]





###############TUNING PARAMETER WITH VALIDATION DATA######################

#parameters:


bias_1=1
best_lambda=3.1
lamda=np.arange(0.1,5,0.3)
gama_1=np.arange(0.1,9,0.1)
cluster_numbers=np.arange(1,40)
best_cluster_number=40
best_gama=0.4
error_list=[]
##########CLUSTER NUMBER TUNING#######
'''
for item in cluster_numbers:
    print(item)
    error_list.append(mse_error(best_gama,item,bias_1,best_lambda)[0])
print(error_list)
plt.plot(cluster_numbers,error_list)
plt.title('number of clusters tuning with validation data')
plt.xlabel('number of clusters')
plt.ylabel('validation error')
plt.show()
'''

#########GAMA TUNING#########
'''
for item in gama_1:
    print(item)
    error_list.append(mse_error(item,best_cluster_number,bias_1,best_lambda)[0])
print(error_list)
plt.plot(gama_1,error_list)
plt.title('gama tuning with validation data')
plt.xlabel('gama')
plt.ylabel('validation error')
plt.show()
'''
##########LAMBDA TUNING######
#########GAMA TUNING#########
'''
for item in lamda:
    print(item)
    error_list.append(mse_error(best_gama,best_cluster_number,bias_1,item)[0])
print(error_list)
plt.plot(lamda,error_list)
plt.title('lambda tuning with validation data')
plt.xlabel('lambda')
plt.ylabel('validation error')
plt.show()
'''
########################CURVE FITTING TO ORIGINAL DATA##############

predict=mse_error(best_gama,best_cluster_number,bias_1,best_lambda)[1]

plt.scatter(total_data[:,0],total_data[:,1], label='Original Data')
plt.scatter(predict[:,0],predict[:,1], label='Fitting Curve')
plt.title('curve fitting to original data')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend(loc='best')
plt.show()
train_mse_error=mse_error(best_gama,best_cluster_number,bias_1,best_lambda)[2]
test_mse_error=mse_error(best_gama,best_cluster_number,bias_1,best_lambda)[3]
print('TRAIN MSE ERROR :')
print(train_mse_error)
print('TEST MSE ERROR :')
print(test_mse_error)

