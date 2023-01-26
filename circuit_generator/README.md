## Quantum Circuit Generator

A random sampled gate quantum circuit generator. Currently, only 3 gates are chosen, hadamard, cnot and phase gates. 

To run the simulations:

`python3 circuit_builder.py {num of qubits} {num of gates per qubit} {num of gaussian samples}`

I created some tests for make sure that the binary tree approach to calculate the conditional probability was correct.

You can run the tests with: 

`pytest cond_prob_test.py`



