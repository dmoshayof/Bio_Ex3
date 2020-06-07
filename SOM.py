import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics.pairwise import euclidean_distances
import math
import random
MATRIX_SIZE = 10
path = 'Digits_Ex3.txt'
NN_SIZE = 6
EPOCHS = 72
ALPHA = 0.9
SIGMA = 0.5
class Neuron():

    def __init__(self,N):
        K = np.random.randint(0,100)
        self.vector = np.zeros(N)
        self.vector[:K] = 1
        np.random.shuffle(self.vector)
        self.color = None

    def set_colors(self,color):
        self.color = color

    def update(self,vector):
        return

class Board():

    def __init__(self,N):
        self.net_work =[[0]*NN_SIZE for i in range(NN_SIZE)]
        for i in range(NN_SIZE):
            random.seed(random.randint(0,100))
            for j in range(NN_SIZE):
                self.net_work[i][j] = np.random.randint(0,2,[MATRIX_SIZE,MATRIX_SIZE])
                np.random.shuffle(self.net_work[i][j])

    def calc_dist(self,i_matrix,cell):
        sum = 0
        for i in range(MATRIX_SIZE):
            for j in range(MATRIX_SIZE):
                sum += pow(int(i_matrix[i][j])-cell[i][j],2)
        total = np.sqrt(sum)
        return total

    def update(self,i,j, i_matrix,alpha,sigma):
        for x in range(NN_SIZE):
            for y in range(NN_SIZE):
                new_row = []
                k =0
                for row in self.net_work[x][y]:
                    dist = math.sqrt((i-x)**2+(j-y)**2)
                    h = alpha *(np.exp(-dist/(2*(sigma**2))))
                    new_row.append(row+(h*(i_matrix[k]-row)))
                    k +=1
                self.net_work[x][y] = new_row
        return
    def convert_to_matrix(self,sample):
        matrix = np.zeros([10, 10])
        num = sample.split('\n')
        for i in range(0, 10):
            for j in range(0, 10):
                matrix[i][j] = num[i][j]
        return matrix
    def show_image(self):
        # subplot(r,c) provide the no. of rows and columns
        f, axarr = plt.subplots(6,6)
        for i in range(NN_SIZE):
            for j in range(NN_SIZE):
             axarr[i][j].imshow(self.net_work[j][i])
        [axi.set_axis_off() for axi in axarr.ravel()]
        plt.waitforbuttonpress()
        plt.close()

    def train(self,data):
        dist = {}
        t_alpha = ALPHA
        t_sigma = SIGMA


        for e in range(EPOCHS):
            if e == 20:
                np.random.shuffle(data)
            for sample in data:
                i_matrix = self.convert_to_matrix(sample)
                for i in range(NN_SIZE):
                    for j in range(NN_SIZE):
                        dist[(i,j)]=(self.calc_dist(i_matrix,self.net_work[i][j]))
                min_item = min(dist.items(), key=lambda x: x[1])
                n_row = min_item[0][0]
                n_col = min_item[0][1]
                self.update(n_row,n_col,i_matrix,t_alpha,t_sigma)
            t_alpha = ALPHA*(t_alpha/ALPHA)**(e/EPOCHS)
            t_sigma = SIGMA*(t_sigma/SIGMA)**(e/EPOCHS)

        self.show_image()




def main():
    with open(path) as data:
        all_num = data.read().split('\n\n')

    board = Board(N=100)
    board.train(all_num)


if __name__ == '__main__':
    digits = load_digits()
    data = digits.data
    main()
