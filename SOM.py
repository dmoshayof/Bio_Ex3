import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

MATRIX_SIZE = 10
path = 'Digits_Ex3.txt'
NN_SIZE = 6
class Neuron():

    def __init__(self,K,N):
        self.vector  = np.zeros(N)
        self.vector[:K] = 1
        np.random.shuffle(self.vector)
        self.color = None

    def update(self,vector):
        return

class Board():

    def __init__(self,K,N):
        self.net_work = [[Neuron(K,N)]*NN_SIZE]*NN_SIZE

    def calc_dist(self,new_vec,neuron):
        sum = 0
        for i in range(0,np.size(neuron.vector)):
            sum += pow(int(new_vec[i])-neuron.vector[i],2)
        total = np.sqrt(sum)
        return total

    def train(self,data):
        dist = {}
        for sample in data:
            sample = sample.replace('\n','')
            for row in self.net_work:
                for i,neuron in enumerate(row):
                    dist[neuron]=(self.calc_dist(sample,neuron))


def main():
    with open(path) as data:
        all_num = data.read().split('\n\n')

    matrix = np.zeros([10,10])
    for num in all_num:
        num = num.split('\n')
        for i in range(0,10):
            for j in range(0,10):
                matrix[i][j] =num[i][j]
        im = plt.imshow(matrix)
        #plt.pause(0.01)
    board = Board(36,100)
    board.train(all_num[0].replace('\n',''))


if __name__ == '__main__':
    digits = load_digits()
    data = digits.data
    main()
