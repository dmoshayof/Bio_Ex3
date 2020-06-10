import numpy as np
import matplotlib.pyplot as plt
import math
import random

MATRIX_SIZE = 10
path = 'Digits_Ex3.txt'
NN_SIZE = 6
EPOCHS = 1
ALPHA = 0.9
SIGMA = 0.4

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np; np.random.seed(42)



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
    def convert_to_matrix(data):
        all_matrix = []
        all_matrix = np.empty((100, 10, 10))
        row = 0
        for sample in data:
            matrix = np.zeros([10, 10])
            num = sample.split('\n')

            for i in range(0, 10):
                for j in range(0, 10):
                    matrix[i][j] = num[i][j]
            all_matrix[row] = matrix
            row+=1

        return all_matrix

    def show_image(self,data):
        # subplot(r,c) provide the no. of rows and columns
        f, axarr = plt.subplots(16, 16)
        for i in range(NN_SIZE):
            for j in range(NN_SIZE):
                axarr[i][j].imshow(self.net_work[j][i])
        count = 0

        for i in range(MATRIX_SIZE):
            for j in range(MATRIX_SIZE):
                axarr[i+6][j+6].imshow(data[count][0])
                axarr[i+6][j+6].set_title(data[count][1])
                count+=1

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

    def train(self, all_matrix):
        save = []

        t_alpha = ALPHA
        t_sigma = SIGMA
        for e in range(EPOCHS):
            #np.random.shuffle(data)  #For data shuffling state
            avg_quantization_error = 0
            for i_matrix in all_matrix:
                #sample = random.choice(data) #for random data state
                n_row, n_col = self.find_nearest_neuron(i_matrix)
                self.update(n_row, n_col, i_matrix, t_alpha, t_sigma)
                avg_quantization_error += self.calc_dist(i_matrix, self.net_work[n_row][n_col])
                if e == EPOCHS -1:
                    save.append((i_matrix,(n_row,n_col)))
            avg_quantization_error /= MATRIX_SIZE*MATRIX_SIZE
            print("Epoch: {} , quantization error: {}".format(e, avg_quantization_error))
            t_alpha = ALPHA * (t_alpha / ALPHA) ** (e / EPOCHS)
            t_sigma = SIGMA * (t_sigma / SIGMA) ** (e / EPOCHS)
        self.show_image(save)
        return save

# Generate data x, y for scatter and an array of images.
x = np.arange(6)
y = np.arange(6)

with open(path) as data:
    all_num = data.read().split('\n\n')
som = SOM()
arr = som.convert_to_matrix(all_num)
#som.train(arr)
# create figure and plot scatter
fig = plt.figure()
ax = fig.add_subplot(111)
line, = ax.plot(x,y, ls="", marker="o")

# create the annotations box
im = OffsetImage(arr[0,:,:], zoom=5)
xybox=(50., 50.)
ab = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',
        boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))
# add it to the axes and make it invisible
ax.add_artist(ab)
ab.set_visible(False)

def hover(event):
    # if the mouse is over the scatter points
    if line.contains(event)[0]:
        # find out the index within the array from the event
        ind, = line.contains(event)[1]["ind"]
        # get the figure size
        w,h = fig.get_size_inches()*fig.dpi
        ws = (event.x > w/2.)*-1 + (event.x <= w/2.)
        hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
        # if event occurs in the top or right quadrant of the figure,
        # change the annotation box position relative to mouse.
        ab.xybox = (xybox[0]*ws, xybox[1]*hs)
        # make annotation box visible
        ab.set_visible(True)
        # place it at the position of the hovered scatter point
        ab.xy =(x[ind], y[ind])
        # set the image corresponding to that point
        im.set_data(arr[ind,:,:])
    else:
        #if the mouse is not over a scatter point
        ab.set_visible(False)
    fig.canvas.draw_idle()

# add callback for mouse moves
fig.canvas.mpl_connect('motion_notify_event', hover)
plt.show()
def main():
    with open(path) as data:
        all_num = data.read().split('\n\n')

    som = SOM()
    som.train(all_num)


if __name__ == '__main__':
    main()
