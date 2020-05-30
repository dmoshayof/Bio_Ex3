import numpy as np
import matplotlib.pyplot as plt

MATRIX_SIZE = 10
path = 'Digits_Ex3.txt'
NN_SIZE = 6
class Board():

    def rand_bin_matrix(self,K, N):
        self.net_work = np.zeros([6,6,N])
        for i, row in enumerate(self.net_work):
            for j,cell in enumerate(row):
                arr = np.zeros(N)
                arr[:K] = 1
                np.random.shuffle(arr)
                self.net_work[i][j] = arr

    def calc_dist(self,new_vec,old_vec):
        sum = 0
        for i in range(0,np.size(old_vec)):
            sum += pow(int(new_vec[i])-old_vec[i],2)
        total = np.sqrt(sum)
        return total

    def train(self,data):
        dist = 0
        for sample in data:
            sample = sample.replace('\n','')
            for row in self.net_work:
                for i,cell in enumerate(row):
                    self.calc_dist(sample,cell)


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
        plt.pause(0.01)
    board = Board()
    board.rand_bin_matrix(36,100)
    board.train(all_num[0].replace('\n',''))


if __name__ == '__main__':
    main()
