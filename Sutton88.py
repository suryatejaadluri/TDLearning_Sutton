import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
#True weights of the random walk model.
expected_w = np.array([1/6., 1/3., 1/2., 2/3., 5/6.])

#Generating test cases using uniform distribution random walk
def step(state):
    chance = np.random.uniform(0,1.0)
    if chance>=0.5:
        state = state+1
    else:
        state = state-1
    return state

def walk():
    random_path = []
    initial_state = 3
    random_path.append(initial_state)
    state = step(initial_state)
    random_path.append(state)
    while state!=6 and state!=0:
        state = step(state)
        random_path.append(state)
    return np.array(random_path)

def gen_trainingset():
    np.random.seed(56)
    n_trainingsets = 100
    n_sequences = 10
    trainingsets = []
    for i in range(n_trainingsets):
        trainingset = []
        for j in range(n_sequences):
            sequence = walk()
            trainingset.append(sequence)
        trainingsets.append(trainingset)
    return trainingsets

# Method for One-hot encoding the generated sequences
def get_vectorsequence(array):
    seq_vectors = []
    for state in array:
        if state == 0 or state == 6:
            if state == 0:
                seq_vectors.append(0)
            else:
                seq_vectors.append(1)
            break
        vector = np.zeros(5)
        if vector[state-1] == 0:
            vector[state-1] = 1
            seq_vectors.append(vector)
    return seq_vectors

#Experiment 1 - Batch presentation
def first_experiment(seed,alpha,lambda_values):
    np.random.seed(seed)
    weights = np.array(np.random.uniform(0,1,5))
    lambda_values = lambda_values
    alpha = alpha
    training_set = gen_trainingset()
    training_set = np.array(training_set,dtype=object)

    #finalws_lambda consists of weights of 100 training sets for each lambda
    final_ws_lambda = []
    for lamda in lambda_values:
        #final_ws_set consists of final converged weights of each traning set
        final_ws_set = []
        for each_set in training_set:
            #initializing the w vector for this training set with initial weights
            w = np.array(weights)
            while True:
                #delta_ws_seq consisits of each sequences final weight vector
                delta_ws_seq = []
                for each_sequence in each_set:
                    sequence = get_vectorsequence(each_sequence)
                    e_t = sequence[0]
                    for t in range(len(sequence)-1):
                        P_t = np.dot(sequence[t],w)
                        if t == len(sequence) - 2:
                            P_t1 = sequence[t+1]
                        else:
                            P_t1 = np.dot(sequence[t+1],w)
                        delta_wt = (alpha * (P_t1-P_t)) * e_t
                        delta_ws_seq.append(delta_wt)
                        # update e incrementally
                        e_t1 = sequence[t+1]+(lamda * e_t)
                        e_t = e_t1
                delta_w = np.sum(delta_ws_seq, axis=0)
                old_w = deepcopy(w)
                w += delta_w
                diff = np.abs(old_w - w)
                if np.sum(diff) < 0.001:
                    final_ws_set.append(w)
                    break
        final_ws_lambda.append(final_ws_set)
    final_ws_lambda = np.array(final_ws_lambda,dtype=object)
    return final_ws_lambda

#Experiment 2- Learning rates and Limited data
def second_experiment(alpha,lambda_values):
    weights = [0.5,0.5,0.5,0.5,0.5]
    lambda_values = lambda_values
    alpha = alpha
    training_set = gen_trainingset()
    training_set = np.array(training_set, dtype=object)

    # finalws_lambda consists of weights of 100 training sets for each lambda (7,100,5)
    final_ws_lambda = []
    for lamda in lambda_values:
        # final_ws_set consists of final weights of each traning set (100(sigma10seq),5)
        final_ws_set = []
        for each_set in training_set:
            # initializing the w vector for this training set with initial weights
            w = np.array(weights)
            for each_sequence in each_set:
                # delta_ws_seq consisits of each sequence's weight vector (seq(length),5)
                delta_ws_seq = []
                sequence = get_vectorsequence(each_sequence)
                e = sequence[0]
                for t in range(len(sequence)-1):
                    P_t = np.dot(sequence[t], w)
                    if t == len(sequence) - 2:
                        P_t1 = sequence[t+1]
                    else:
                        P_t1 = np.dot(sequence[t+1],w)
                    delta_wt = (alpha * (P_t1-P_t)) * e
                    delta_ws_seq.append(delta_wt)
                    # update e incrementally
                    e_t1 = sequence[t+1] + (lamda * e)
                    e = e_t1
                delta_w = np.sum(delta_ws_seq, axis=0)
                w += delta_w
            final_ws_set.append(w)
        final_ws_lambda.append(final_ws_set)
    final_ws_lambda = np.array(final_ws_lambda, dtype=object)
    return final_ws_lambda

#Function to calculate RMS error from first experiment and to generate Figure 3
def generate_fig3():
    lambda_values = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    results = first_experiment(55,0.01,lambda_values)
    #Calculating RMS error
    error = ((results - expected_w)**2) #numerator
    z = np.sum(error,axis=2)/5 #mean
    z = np.array(z,dtype=float)
    z = np.sqrt(z) #square root
    y = np.mean(z,axis = 1) #avg over 100 training sets
    plt.plot(lambda_values,y,marker = 'o', ms = 5 )
    plt.xlabel("\u03bb")
    plt.ylabel("Error")
    plt.title('Fig.3 TD(\u03bb) performance for different \u03bb values')
    plt.xticks(lambda_values,lambda_values)
    plt.show()

#Function to calculate RMS error from second experiment and to generate Figure 4
def generate_fig4():
    lambda_values = [0, 0.3, 0.8, 1]
    alpha_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    alpha_errors = []
    for alpha in alpha_values:
        results = second_experiment(alpha, lambda_values)
        #calculating RMS error
        error = ((results - expected_w) ** 2)  # numerator
        z = np.sum(error, axis=2) / 5  # mean
        z = np.array(z, dtype=float)
        z = np.sqrt(z)  # square root
        y = np.mean(z, axis=1)  # avg over 100 training sets
        alpha_errors.append(y)

    plt.plot(alpha_values, alpha_errors, marker = 'o', ms = 5)
    plt.xlabel("learning rate(alpha)")
    plt.ylabel("RMS Error")
    plt.title('Fig.4 TD(\u03bb) performance concerning learning rates')
    plt.legend(['\u03bb 0','\u03bb 0.3', '\u03bb 0.8', '\u03bb 1'])
    plt.ylim(0.0,1.1)
    plt.show()

#Function to calculate RMS error from second experiment and to generate Fifure 5
def generate_fig5():
    lambda_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1]
    alpha_values = [0,0.1,0.2,0.3,0.4,0.5,0.6]
    alpha_errors = []
    for alpha in alpha_values:
        results = second_experiment(alpha, lambda_values)
        #calculating RMS error
        error = ((results - expected_w) ** 2)  # numerator
        z = np.sum(error, axis=2) / 5  # mean
        z = np.array(z, dtype=float)
        z = np.sqrt(z)  # square root
        y = np.mean(z, axis=1)  # avg over 100 training sets
        alpha_errors.append(y)

    alpha_best = np.min(alpha_errors,axis=0)
    plt.plot(lambda_values, alpha_best, marker = 'o', ms = 5)
    plt.xlabel("\u03bb")
    plt.ylabel("RMS Error using the best alpha")
    plt.title(' Fig.5 TD(\u03bb) performance using the best learning rates')
    plt.ylim(0.075,0.19)
    plt.show()

#Shows the weight updates for each training set for TD(0)
def generate_fig6(alpha = 0.6,lamda = 0):
    lambda_values = [lamda]
    alpha_values = [alpha]
    weights = expected_w
    x_plot = []
    for alpha in alpha_values:
        results = second_experiment(alpha, lambda_values)
        plot_weights_TD = results[0]
        x_plot.append(plot_weights_TD.T)

    plt.plot(weights, x_plot[0], marker = 'o', ms = 5)
    plt.xlabel("Weight Vector")
    plt.ylabel("Converged Weights")
    plt.title('Fig.6 Shows the weight updates for each training set for TD(0)')
    plt.show()

#Shows the weight updates for each training set for TD(1)
def generate_fig7(alpha = 0.6,lamda = 1):
    lambda_values = [lamda]
    alpha_values = [alpha]
    weights = expected_w
    x_plot = []
    for alpha in alpha_values:
        results = second_experiment(alpha, lambda_values)
        plot_weights_TD = results[0]
        x_plot.append(plot_weights_TD.T)

    plt.plot(weights, x_plot[0], marker = 'o', ms = 5)
    plt.xlabel("Weight Vector")
    plt.ylabel("Converged Weights")
    plt.title('Fig.7 Shows the weight updates for each training set for TD(1)')
    plt.show()

#Shows the weight updates for each training set for TD(0.8)
def generate_fig8(alpha = 0.6 ,lamda = 0.8):
    lambda_values = [lamda]
    alpha_values = [alpha]
    weights = expected_w
    x_plot = []
    for alpha in alpha_values:
        results = second_experiment(alpha, lambda_values)
        plot_weights_TD = results[0]
        x_plot.append(plot_weights_TD.T)

    plt.plot(weights, x_plot[0], marker = 'o', ms = 5)
    plt.xlabel("Weight Vector")
    plt.ylabel("Converged Weights")
    plt.title('Fig.8 Shows the weight updates for each training set for TD(0.8)')
    plt.show()

if __name__ == '__main__':
    globals()[sys.argv[1]]()
