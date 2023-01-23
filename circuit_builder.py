import tequila as tq
import numpy as np
import numpy.random as npr
from tequila import gates as tq_g
import matplotlib.pyplot as plt
from scipy.stats import norm
import random

class CircuitBuilder():
    def __init__(self, qubits, max_params):
        self.circuit = tq.QCircuit()
        self.qubits = qubits
        self.max_params = max_params
        self.H = np.zeros((qubits, max_params))
        self.S = np.zeros((qubits, max_params))
        self.CNOT = np.zeros((qubits, max_params))
        self.wfn = None

    def add_hadamard(self, index, qubit_num):
        self.circuit += tq_g.H(qubit_num)
        self.H[qubit_num, index] = 1
    
    def add_phase(self, index, qubit_num):
        self.circuit += tq_g.S(qubit_num)
        self.S[qubit_num, index] = 1
    
    def add_cnot(self, index, qubit1, qubit2):
        self.circuit +=tq_g.CNOT(qubit1, qubit2)
        self.CNOT[qubit1, index] = 1

    
    def gen_gates(self, len_gates):
        gate_index = np.random.randint(0, len_gates, (self.max_params, self.qubits))
        cnot_qubit_selector = np.zeros((self.max_params, self.qubits), dtype=np.int8)

        for i in range(self.max_params):
            for j in range(self.qubits):
                if gate_index[i, j] == 2:
                    cnot_qubit_selector[i, j] = random.choice(list(set([x for x in range(0, self.qubits)]) - set([j])))
        
        return gate_index, cnot_qubit_selector

    
    def create(self):
        gates = [self.add_hadamard, self.add_phase, self.add_cnot]

        gate_index, cnot_selector = self.gen_gates(len(gates))

        for i in range(self.max_params):
            for j in range(self.qubits):
                    gate_num = gate_index[i, j]
                    if gate_num == 2:
                        gates[2](i, j, cnot_selector[i, j])
                    else:
                        gates[gate_num](i, j)


    def gen_y(self):
        y = np.vstack((self.H, self.S, self.CNOT))
        reshape_y = np.reshape(y, (3, self.qubits, self.max_params))
        return reshape_y
    
    def gen_x_full_gaussian(self, num_samples):
        x = np.zeros((self.qubits, num_samples))
        cond_probs = np.asarray(self.get_con_prob()).clip(0.08, 0.92)
        for i, prob in enumerate(cond_probs):
            denom = norm.ppf(1 - (prob / 2))
            sig = 1 / denom
            x[i] = npr.normal(0, sig, num_samples)
        
        return x
    
    def gen_x_half_gaussian(self, num_samples):
        x = np.zeros((self.qubits, num_samples))
        cond_probs = np.asarray(self.get_con_prob()).clip(0.04, 0.96)
        for i, prob in enumerate(cond_probs):
            denom = norm.ppf(1 - (prob / 2))
            sig = 1 / denom
            # pad some samples so we always have num samples positive values
            temp = npr.normal(0, sig, (2 * (num_samples + int(1 / 5 * num_samples))))
            half = np.delete(temp, np.where(temp < 0))
            #remove from tails of the gaussian
            x[i] = half[:num_samples]
            assert(x.shape[1] == num_samples)

        return x
    
    def gen_dataset_sample(self, num_samples, xtype='half'):
        y = self.gen_y()

        if xtype == 'half':
            x = self.gen_x_half_gaussian(num_samples)
        else:
            x = self.gen_x_full_gaussian(num_samples)

        return x, y

    def draw_circuit(self):
        tq.draw(self.circuit, backend="cirq")
        print(f'H: \n {self.H}\n')
        print(f'S: \n {self.S}\n')
        print(f'CNOT: \n {self.CNOT}\n')

    
    def get_wfn(self, samples=None):
        initial_param = {var:0.1 for var in self.circuit.extract_variables()}
        wfn = tq.simulate(self.circuit, variables=initial_param, samples=None, backend='qulacs')
        self.wfn = wfn
        return wfn
    
    def get_prob(self, samples=None):
        self.get_wfn(samples=samples)
        probs = np.zeros(2 ** self.qubits, dtype=complex)
        for i in range(self.qubits ** 2):
            if i not in self.wfn._state:
                continue 
            else:
                probs[i] = self.wfn._state[i]

        probs = np.power(np.abs(probs), 2)
        return probs

    
    def get_con_prob(self):
        # splitting bits to calculate cond probs can be seen as a binary tree problem
        # we solve for the conditional probabilties of 0s on each qubit
        # for example with 3 qubits: |1~~>, |~1~>, |~~1> 

        probs = self.get_prob()

        def recur(lst):
            # the 2 qubit is our base case
            # we get conditionl probabilities of |1~>, |~1>
            if len(lst) == 4:
                return [lst[1] + lst[3], lst[2] + lst[3]]
            else:
                # split bits, should be divisible by 2
                mid = len(lst) // 2
                left = recur(lst[:mid])
                right = recur(lst[mid:])

                temp = []

                for i in range(len(right)):
                    temp.append(left[i] + right[i])
                temp.append(sum(lst[mid:]))

                return temp

        cond_probs = recur(probs)
        return cond_probs

    def get_expect(self, samples=None):
        ham = tq.paulis.Z(0)*tq.paulis.Z(1)#*tq.paulis.Z(2)*tq.paulis.Z(3)

        #initialize parameters dictionary again
        initial_param = {var:0.1 for var in self.circuit.extract_variables()}

        #creating the Expectation Value object
        Expval = tq.ExpectationValue(H=ham, U=self.circuit)
        sampled_wfn = tq.simulate(Expval, variables=initial_param, samples=samples, backend='qulacs')
        return sampled_wfn
    
if __name__ == "__main__":
    num_qubits = 3
    cb = CircuitBuilder(num_qubits, 10)
    cb.create()
    wfn = cb.get_wfn()
    probs = cb.get_prob()
    print(probs)
    cond_probs = cb.get_con_prob()
    print(cond_probs)
    test = cb.gen_x(1000)
        

    for i, dis in enumerate(test):
        plt.figure()
        plt.hist(dis, bins=100)
        plt.savefig(f"qubit_{i}")
        plt.close()
        plt.xlim(-5, 5)
        test1 = (np.where((dis >= 1) | (dis <= -1), 1, 0))
        print(sum(test1) / 10000)











        