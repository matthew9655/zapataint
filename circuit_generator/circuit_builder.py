import tequila as tq
import numpy as np
import numpy.random as npr
from tequila import gates as tq_g
from scipy.stats import norm
import random
import sys

class CircuitBuilder():
    def __init__(self, qubits, max_gates):
        self.circuit = tq.QCircuit()
        self.qubits = qubits
        self.max_gates = max_gates
        self.H = np.zeros((qubits, max_gates))
        self.S = np.zeros((qubits, max_gates))
        self.CNOT = np.zeros((qubits, max_gates))
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
        "randomly sample the gates to add to the circuit"
        gate_index = np.random.randint(0, len_gates, (self.max_gates, self.qubits))
        cnot_qubit_selector = np.zeros((self.max_gates, self.qubits), dtype=np.int8)

        for i in range(self.max_gates):
            for j in range(self.qubits):
                if gate_index[i, j] == 2:
                    cnot_qubit_selector[i, j] = random.choice(list(set([x for x in range(0, self.qubits)]) - set([j])))
        
        return gate_index, cnot_qubit_selector

    
    def create(self):
        "create the circuit"
        gates = [self.add_hadamard, self.add_phase, self.add_cnot]

        gate_index, cnot_selector = self.gen_gates(len(gates))

        for i in range(self.max_gates):
            for j in range(self.qubits):
                    gate_num = gate_index[i, j]
                    if gate_num == 2:
                        gates[2](i, j, cnot_selector[i, j])
                    else:
                        gates[gate_num](i, j)


    def gen_y(self):
        "concatenate the matrix representation of the circuit for conditional input to Masked Autoregressive Flow"
        y = np.vstack((self.H, self.S, self.CNOT))
        reshape_y = np.reshape(y, (3, self.qubits, self.max_gates))
        return reshape_y
    
    def gen_x_full_gaussian(self, num_samples):
        "generate dataset for full gaussian representations"
        x = np.zeros((self.qubits, num_samples))
        cond_probs = np.asarray(self.get_con_prob()).clip(0.08, 0.92)
        for i, prob in enumerate(cond_probs):
            denom = norm.ppf(1 - (prob / 2))
            sig = 1 / denom
            x[i] = npr.normal(0, sig, num_samples)
        
        return x
    
    def gen_x_half_gaussian(self, num_samples):
        "generate dataset for half gaussian representations"
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
        "calculate the wavefunction of the circuit "
        initial_param = {var:0.1 for var in self.circuit.extract_variables()}
        wfn = tq.simulate(self.circuit, variables=initial_param, samples=None, backend='qulacs')
        self.wfn = wfn
        return wfn
    
    def get_prob(self, samples=None):
        "calculate the probability of each measurement outcome"
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
        """
        splitting bits to calculate cond probs can be seen as a binary tree problem
        we solve for the conditional probabilties of 0s on each qubit
        for example with 3 qubits: |1~~>, |~1~>, |~~1> 
        """
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
    
if __name__ == "__main__":
    num_qubits = int(sys.argv[1])
    max_gates = int(sys.argv[2])
    num_samples = int(sys.argv[3])
    cb = CircuitBuilder(num_qubits, max_gates)
    cb.create()
    print('probability of each measurement outcome e,g 000, 001 ...')
    probs = cb.get_prob()
    print(probs)
    print('conditional probability of each individual qubit being state 1 in the z basis')
    cond_probs = cb.get_con_prob()
    print(cond_probs)
    x, y = cb.gen_dataset_sample(num_samples)
        
    for i, dis in enumerate(x):
        test1 = (np.where((dis >= 1) | (dis <= -1), 1, 0))
        print(f"qubit-{i}: {sum(test1) / num_samples}")











        