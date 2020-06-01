import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

MATRIX_SIZE = 10
path = 'Digits_Ex3.txt'
NN_SIZE = 6
class Neuron():

    def __init__(self,N):
        K = np.random.randint(0,100)
        self.vector = np.zeros(N)
        self.vector[:K] = 1
        np.random.shuffle(self.vector)
        self.color = None

    def update(self,vector):
        return

class Board():

    def __init__(self,N):
        self.net_work = [[]*NN_SIZE]*NN_SIZE
        for i in range(0,NN_SIZE):
            self.net_work[i].append( Neuron(N))




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
            for i,row in enumerate(self.net_work):
                for j,neuron in enumerate(row):
                    dist[(i,j)]=(self.calc_dist(sample,neuron))
            min_item = min(dist.items(), key=lambda x: x[1])
            n_row = min_item[0][0]
            n_col = min_item[0][1]
            self.net_work[n_row][n_col].update(sample)



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
    board = Board(N=100)
    board.train(all_num)


if __name__ == '__main__':
    digits = load_digits()
    data = digits.data
    main()
