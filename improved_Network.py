import random
import numpy as np
import loader
import json

class CrossEntropyCost(object):

    @staticmethod
    def cost_fn(output_activations, y) :
        return np.sum(np.nan_to_num(-y*np.log(output_activations) -(1 - y)*np.log(1 - output_activations)))

    @staticmethod
    def delta(z, output_activations, y):
        return (output_activations - y)

class QuadraticCost(object):

    @staticmethod
    def cost_fn(output_activations, y) :
        return 0.5*np.linalg.norm(output_activations - y)**2

    @staticmethod
    def delta(z, output_activations, y):
        return (output_activations - y)*sigmoid_prime(z)

class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost, weight_initializer='default'):
        
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = self.default_weight_initializer()
        if weight_initializer == 'large':
            self.weights = self.large_weight_initializer()

    def default_weight_initializer(self):
        return [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
    
    def large_weight_initializer(self) :
        return [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):

        training_data = list(training_data)
        n = len(training_data)
        if evaluation_data is not None :
            evaluation_data = list(evaluation_data)
            n_evaluation = len(evaluation_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n)
            
            print("\nEpoch %s training complete" % j)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data : {}".format(cost))
            if monitor_training_accuracy:
                correct = self.evaluate(training_data)
                training_accuracy.append(100.0*correct/n)
                print("Accuracy on training data : {0}/{1}".format(correct, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda)
                evaluation_cost.append(cost)
                print("Cost on validation data : {}".format(cost))
            if monitor_evaluation_accuracy:
                correct = self.evaluate(evaluation_data)
                evaluation_accuracy.append(100.0*correct/n_evaluation)
                print("Accuracy on validation data : {0}/{1}".format(correct, n_evaluation))
            
        return training_cost, training_accuracy, evaluation_cost, evaluation_accuracy


    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1 - eta*(lmbda/n))*w - (eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]

        self.biases = [b - (eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return (nabla_b, nabla_w)

    def total_cost(self, data, lmbda):
        cost = 0.0
        n = len(data)
        
        for x, y in data :
            a = self.feedforward(x)
            cost += self.cost.cost_fn(a, y)/n
        cost += 0.5*(lmbda/n)*sum(np.linalg.norm(w)**2 
                                    for w in self.weights)

        return cost

    def evaluate(self, data):
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in test_results)

    def save(self, filename):
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

if __name__ == '__main__':
    tr, va, te = loader.load_data_wrapper()
    network = Network([784, 30, 10])
    network.SGD(tr, 30, 10, 0.5, 5.0, va, True, True, True, True)
    network.save("improved_network_results.json")