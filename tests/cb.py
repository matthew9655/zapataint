import pytest
from ..circuit_builder import CircuitBuilder 
from collections import defaultdict

def gen_index_dic(num_qubit):
    dic = defaultdict(list)

    for i in range(2 ** num_qubit):
        binned = format(i, f'0{num_qubit}b')
        for j in range(num_qubit):
            if binned[j] == '1':
                dic[num_qubit-1-j].append(i)
    
    return dic
        

def test_3_qubit_cond_prob():

    def manual_cond_prob(lst):
        cond_probs = [0] * 3

        cond_probs[0] = lst[1] + lst[3] + lst[5] + lst[7]
        cond_probs[1] = lst[2] + lst[3] + lst[6] + lst[7]
        cond_probs[2] = lst[4] + lst[5] + lst[6] + lst[7]

        return cond_probs

    for _ in range(100):
        cb = CircuitBuilder(3, 10)
        cb.create()
        probs = cb.get_prob()
        ans = manual_cond_prob(probs)
        algo = cb.get_con_prob()

        assert pytest.approx(algo, 0.01) == ans

def test_4_qubit_cond_prob():

    def manual_cond_prob(lst):
        cond_probs = [0] * 4

        dic = gen_index_dic(4)

        for i in range(4):
            for j in range(8):
                cond_probs[i] += probs[dic[i][j]]

        return cond_probs

    for _ in range(100):
        cb = CircuitBuilder(4, 10)
        cb.create()
        probs = cb.get_prob()
        ans = manual_cond_prob(probs)
        algo = cb.get_con_prob()

        assert pytest.approx(algo, 0.01) == ans


def test_5_qubit_cond_prob():

    def manual_cond_prob(lst):
        cond_probs = [0] * 5

        dic = gen_index_dic(5)

        for i in range(5):
            for j in range(16):
                cond_probs[i] += probs[dic[i][j]]

        return cond_probs

    for _ in range(100):
        cb = CircuitBuilder(5, 10)
        cb.create()
        probs = cb.get_prob()
        ans = manual_cond_prob(probs)
        algo = cb.get_con_prob()

        assert pytest.approx(algo, 0.01) == ans
