## Quantum Circuit Generator

A random sampled gate quantum circuit generator. Currently, only 3 gates are chosen, Hadamard, CNOT, and phase gates. I wanted to try to think of ways to keep the exponential measurement outcome linear. So this code calculates the 
conditional probability that a qubit is in the 1 state when measured in the z-basis. I then use this conditional probability to create a gaussian distribution where the probability of a sample being greater than -1 or 1 is equal to that conditional probability.

To run the simulations:

`python3 circuit_builder.py {num of qubits} {num of gates per qubit} {num of gaussian samples}`

I created some tests to make sure that the binary tree approach to calculate the conditional probability was correct.

You can run the tests with: 

`pytest cond_prob_test.py`




