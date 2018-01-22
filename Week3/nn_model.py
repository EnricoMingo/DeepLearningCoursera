import numpy as np
import basic_functions as bf

def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    A2, cache = bf.forward_propagation(X, parameters)
    predictions = np.rint(A2)
    
    return predictions

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    np.random.seed(3)
    n_x = bf.layer_sizes(X, Y)[0]
    n_y = bf.layer_sizes(X, Y)[1]
    
    parameters = bf.initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    for i in range(0, num_iterations):
         
        A2, cache = bf.forward_propagation(X, parameters)
        
        cost = bf.compute_cost(A2, Y, parameters)
 
        grads = bf.backward_propagation(parameters, cache, X, Y)
 
        parameters = bf.update_parameters(parameters, grads, learning_rate = 1.2)
        
        
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters