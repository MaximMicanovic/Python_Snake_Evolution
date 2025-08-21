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

def list_compare(a, b):
    """
    Robust comparison for nested lists/tuples and numpy arrays.
    Returns True if structures have the same shape and element-wise equality.
    Handles numpy.ndarray, Python lists/tuples and scalar values. Avoids
    ambiguous truth-value errors from array comparisons.
    """
    # Handle exact identity / None
    if a is b:
        return True
    if a is None or b is None:
        return False

    # If either is a numpy array, convert to numpy for shape inspection
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        # Try to compare shapes first
        try:
            a_arr = a if isinstance(a, np.ndarray) else np.array(a, dtype=object)
            b_arr = b if isinstance(b, np.ndarray) else np.array(b, dtype=object)
        except Exception:
            return False

        if a_arr.shape != b_arr.shape:
            return False

        # Element-wise recursive comparison (handles nested arrays/lists)
        for idx, (ai, bi) in enumerate(zip(a_arr.flat, b_arr.flat)):
            if isinstance(ai, (list, tuple, np.ndarray)) or isinstance(bi, (list, tuple, np.ndarray)):
                if not list_compare(ai, bi):
                    return False
            else:
                # numpy scalar comparison yields numpy.bool_ which is fine here
                if ai != bi:
                    return False
        return True

    # If both are sequences (list/tuple), compare lengths and recurse
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            if not list_compare(a[i], b[i]):
                return False
        return True

    # Fallback scalar comparison
    return a == b

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

    def __eq__(self, other):
        if not isinstance(other, network):
            return False
        
        return (list_compare(self.size, other.size) and
                list_compare(self.weight, other.weight) and
                list_compare(self.bias, other.bias))

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


    # Numpy used to save the weights, biases and outputs
    def save(self, name):
        """Save weights, biases and outputs to .npy files."""
        w = np.array(self.weight, dtype=object)
        b = np.array(self.bias, dtype=object)
        o = np.array(self.outputs, dtype=object)
        np.save(f"{name}_weight.npy", w)
        np.save(f"{name}_bias.npy", b)
        np.save(f"{name}_outputs.npy", o)

    def load(self, name):
        """Load weights, biases and outputs saved by save()."""
        w = np.load(f"{name}_weight.npy", allow_pickle=True)
        b = np.load(f"{name}_bias.npy", allow_pickle=True)
        o = np.load(f"{name}_outputs.npy", allow_pickle=True)

        # assign loaded values back to the instance
        self.weight = list(w)
        self.bias = list(b)
        self.outputs = list(o)

def _children_per_elite(chosen, amount):
    # infer how many children to attempt per selected elite
    # original code computed int(amount/(amount*chosen)) which simplifies to int(1/chosen)
    if chosen and chosen > 0:
        return max(1, int(1 / chosen))
    return 1

def _survival_probability(idx, base_prob):
    # decreasing survival probability for later elites (keeps original behavior)
    return (idx-base_prob)/base_prob

def _mutate_value(value, perturb_scale=0.01):
    # small perturbation used for mutations
    perturb = (random.random() * 2 - 1) * perturb_scale
    return value + perturb

def _crossover_and_mutate(child, parent, will_mutate, mutation_amount):
    # mix weights and biases from child (elite) and parent, optionally mutating individual elements
    for layer in range(len(child.weight)):
        for perc in range(len(child.weight[layer])):
            # mix weights from child and parent
            mix = None
            # weights
            for w_idx in range(len(child.weight[layer][perc])):
                mix = random.random()
                if will_mutate and random.random() < mutation_amount:
                    child.weight[layer][perc][w_idx] = _mutate_value(child.weight[layer][perc][w_idx], mutation_amount)
                else:
                    c_val = child.weight[layer][perc][w_idx]
                    p_val = parent.weight[layer][perc][w_idx]
                    child.weight[layer][perc][w_idx] = c_val * (1 - mix) + p_val * mix

            # bias (single value per perceptron)
            mix = random.random()
            if will_mutate and random.random() < mutation_amount:
                child.bias[layer][perc] = _mutate_value(child.bias[layer][perc], mutation_amount)
            else:
                c_b = child.bias[layer][perc]
                p_b = parent.bias[layer][perc]
                child.bias[layer][perc] = c_b * (1 - mix) + p_b * mix

    return child

def _create_child_from_elite(elite, chosen_children, mutation_chance, mutation_amount):
    parent = chosen_children[random.randrange(0, len(chosen_children))]

    child = copy.deepcopy(elite)
    will_mutate = (random.random() < mutation_chance)

    return _crossover_and_mutate(child, parent, will_mutate, mutation_amount)

def multiply(chosen_children, amount, chosen, mutation_chance, mutation_amount, chanse_to_survive):
    """
    Robust version of multiply:
    - ensures inputs are valid
    - treats chanse_to_survive <= 0 as 1.0 to avoid producing an empty population
    """
    # normalize inputs
    if chosen_children is None:
        exit ("Error: No chosen children provided for multiplication.")

    new_creatures = []
    konstant = _children_per_elite(chosen, amount)

    idx = 0
    # iterate elites and produce children until desired population reached
    while idx < amount and len(new_creatures) < amount:
        elite = chosen_children[idx]
        surv_prob = _survival_probability(idx, chanse_to_survive)
        
        # If it survives and multiplies
        if random.random() >= surv_prob:
            idx += 1
            continue

        new_creatures.append(copy.deepcopy(elite))
        if len(new_creatures) >= amount: 
            break

        # create additional children for this elite
        children_to_create = max(0, konstant - 1)
        for _ in range(children_to_create):
            if len(new_creatures) >= amount:
                break
            child = _create_child_from_elite(elite, chosen_children, mutation_chance, mutation_amount)
            new_creatures.append(child)
        idx += 1


        # if we've exhausted elites but still need more, fill by random crossover
        if (idx-1) == amount and len(new_creatures) < amount:
            while len(new_creatures) < amount:
                parent_idx = random.randrange(0, len(chosen_children))
                elite_idx = random.randrange(0, len(chosen_children))
                parent = chosen_children[parent_idx]
                elite = chosen_children[elite_idx]
                child = copy.deepcopy(elite)
                will_mutate = (random.random() < mutation_chance)
                child = _crossover_and_mutate(child, parent, will_mutate, mutation_amount)
                new_creatures.append(child)

    # ensure we return exactly 'amount' creatures
    
    print("here", len(new_creatures), "creatures created so far")
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
