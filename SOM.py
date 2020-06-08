import numpy as np
import matplotlib.pyplot as plt
import math
import random

MATRIX_SIZE = 10
path = 'Digits_Ex3.txt'
NN_SIZE = 6
EPOCHS = 70
ALPHA = 0.9
SIGMA = 0.4


class SOM():

    def __init__(self):
        '''
        Initialize the net work in network size with random weights 0-1
        shuffle the randomize weights
        '''
        self.net_work = [[0] * NN_SIZE for i in range(NN_SIZE)]
        for i in range(NN_SIZE):
            random.seed(random.randint(0, 100))
            for j in range(NN_SIZE):
                self.net_work[i][j] = np.random.uniform(0, 1, [MATRIX_SIZE, MATRIX_SIZE])
                np.random.shuffle(self.net_work[i][j])

    def calc_dist(self, i_matrix, cell):
        """
        Calc the euclidean distance between two matrix
        :param i_matrix: current sample
        :param cell: the network cell (neuron)
        :return: the euclidean distance
        """
        sum = 0
        for i in range(MATRIX_SIZE):
            for j in range(MATRIX_SIZE):
                sum += pow(int(i_matrix[i][j]) - cell[i][j], 2)
        total = np.sqrt(sum)
        return total

    def update(self, i, j, i_matrix, alpha, sigma):
        """
        Update the network weights with the current sample and most fit neuron
        :param i: row of neuron
        :param j: column of neuron
        :param i_matrix: current sample
        :param alpha: current alpha
        :param sigma: current sigma
        """
        for x in range(NN_SIZE):
            for y in range(NN_SIZE):
                new_row = []
                k = 0
                for row in self.net_work[x][y]:
                    dist = math.sqrt((i - x) ** 2 + (j - y) ** 2)
                    h = alpha * (np.exp(-dist / (2 * (sigma ** 2))))
                    new_row.append(row + (h * (i_matrix[k] - row)))
                    k += 1
                self.net_work[x][y] = new_row

    @staticmethod
    def convert_to_matrix(sample):
        matrix = np.zeros([10, 10])
        num = sample.split('\n')
        for i in range(0, 10):
            for j in range(0, 10):
                matrix[i][j] = num[i][j]
        return matrix

    def show_image(self):
        # subplot(r,c) provide the no. of rows and columns
        f, axarr = plt.subplots(6, 6)
        for i in range(NN_SIZE):
            for j in range(NN_SIZE):
                axarr[i][j].imshow(self.net_work[j][i])
        [axi.set_axis_off() for axi in axarr.ravel()]
        plt.waitforbuttonpress()
        plt.close()

    def find_nearest_neuron(self, i_matrix):
        """
        Calculate all distances from i_matrix to networks neurons and return the nearest
        :param i_matrix: current sample
        :return: nearest neuron index
        """
        dist = {}
        for i in range(NN_SIZE):
            for j in range(NN_SIZE):
                dist[(i, j)] = (self.calc_dist(i_matrix, self.net_work[i][j]))
        min_item = min(dist.items(), key=lambda x: x[1])
        n_row = min_item[0][0]
        n_col = min_item[0][1]
        return n_row, n_col

    def train(self, data):
        t_alpha = ALPHA
        t_sigma = SIGMA
        for e in range(EPOCHS):
            np.random.shuffle(data)  #For data shuffling state
            avg_quantization_error = 0
            for sample in data:
                #sample = random.choice(data) #for random data state
                i_matrix = self.convert_to_matrix(sample)
                n_row, n_col = self.find_nearest_neuron(i_matrix)
                self.update(n_row, n_col, i_matrix, t_alpha, t_sigma)
                avg_quantization_error += self.calc_dist(i_matrix, self.net_work[n_row][n_col])
            avg_quantization_error /= len(data)
            print("Epoch: {} , quantization error: {}".format(e, avg_quantization_error))
            t_alpha = ALPHA * (t_alpha / ALPHA) ** (e / EPOCHS)
            t_sigma = SIGMA * (t_sigma / SIGMA) ** (e / EPOCHS)
        self.show_image()


def main():
    with open(path) as data:
        all_num = data.read().split('\n\n')

    som = SOM()
    som.train(all_num)


if __name__ == '__main__':
    main()
