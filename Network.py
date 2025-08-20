import numpy as np
import math, random, copy

def sigmoid(x):
    return 1/(1 + np.e**(-x))

def primsig(x):
    return sigmoid(x)*(1-sigmoid(x))

def reLU(val):
    if(val < 0):
        return 0
    else:
        return val

class network:
    def __init__(self, size):
        np.random.seed(random.randrange(0,100))
        self.size = size
        self.pos = None

        self.weight = [np.random.randn(size[i], size[i-1]) for i in range(1, len(size))]
        self.bias = [np.random.randn(size[i]) for i in range(1, len(size))]
        self.outputs = [np.zeros(size[i]) for i in range(1, len(size))]
        self.lr = 0.01

        self.aichoise = None

    def forward(self, inp, acfunc):
        for i in range(len(self.weight[0])):
            self.outputs[0][i] = np.dot(self.weight[0][i], inp)
            self.outputs[0][i] += self.bias[0][i]

            if(acfunc == "sigmoid"):
                self.outputs[0][i] = sigmoid(self.outputs[0][i])
            elif(acfunc == "relu"):
                self.outputs[0][i] = reLU(self.outputs[0][i])

        for i in range(1, len(self.weight)):
            for j in range(len(self.weight[i])):
                self.outputs[i][j] = np.dot(self.weight[i][j], self.outputs[i-1])
                self.outputs[i][j] += self.bias[i][j]

                if(acfunc == "sigmoid"):
                    self.outputs[i][j] = sigmoid(self.outputs[i][j])
                elif(acfunc == "relu"):
                    self.outputs[i][j] = reLU(self.outputs[i][j])

        return self.outputs[len(self.outputs)-1]

    def backprop(self, desired):
        #leaving this for now
        print("under development")

    def save(self, name):
        w = np.array(self.weight, dtype=object)
        b = np.array(self.bias, dtype=object)
        o = np.array(self.outputs, dtype=object)
        np.save(f"{name}_weight.npy", w)
        np.save(f"{name}_bias.npy", b)
        np.save(f"{name}_outputs.npy", o)

    def load(self, name):
        self.weight = np.load(f"{name}_weight.npy", allow_pickle=True)
        self.bias = np.load(f"{name}_weight.npy", allow_pickle=True)
        self.outputs = np.load(f"{name}_outputs.npy", allow_pickle=True)

def multiply(chosen_children, amount, chosen, mutation_chance, mutation_amount, chanse_to_survive):
    new_creatures = []
    chosen_tf = [False]*len(chosen_children)
    konstant = int(amount/(amount*chosen))

    while (len(new_creatures) <= amount):
        for c in range(len(chosen_children)):
            if(random.random() < chanse_to_survive*((1-chanse_to_survive)**c) and chosen_tf[c] == False):
                chosen_tf[c] = True
                #Elite Child
                new_creatures.append(copy.deepcopy(chosen_children[c]))

                for create in range(konstant-1):
                    parent = copy.deepcopy(chosen_children[random.randrange(0, len(chosen_children))])
                    child = copy.deepcopy(chosen_children[c])

                    if(random.random() < mutation_chance):
                        mutate = True
                    else:
                        mutate = False

                    for layer in range(len(child.weight)):
                        for perceptron in range(len(child.weight[layer])):
                            father_or_mother = random.random();
                            for weight in range(len(child.weight[layer][perceptron])):
                                if(mutate and random.random() < mutation_amount):
                                    child.weight[layer][perceptron][weight] = child.weight[layer][perceptron][weight] + (random.random()*2 - 1)*0.01
                                else:
                                    child.weight[layer][perceptron][weight] = child.weight[layer][perceptron][weight]*(1-father_or_mother) + parent.weight[layer][perceptron][weight]*(father_or_mother)

                            if(mutate and random.random() < mutation_amount):
                                child.bias[layer][perceptron] = child.bias[layer][perceptron] + (random.random()*2 - 1)*0.01
                            else:
                                child.bias[layer][perceptron] = child.bias[layer][perceptron]*(1-father_or_mother) + parent.bias[layer][perceptron]*(father_or_mother)

                    new_creatures.append(child)

    #Returning the list of new creatures
    return new_creatures


def difference(c1, c2):
    total_size = 0
    same = 0
    for layer in range(len(c1.weight)):
        for perceptron in range(len(c1.weight[layer])):
            for weight in range(len(c1.weight[layer][perceptron])):
                total_size += 1
                if(c1.weight[layer][perceptron][weight] == c2.weight[layer][perceptron][weight]):
                    same += 1

    return same/total_size
