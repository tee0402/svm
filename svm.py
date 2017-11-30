from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
import random


def random_stocks(n=10):
    filename = 'stocks.txt'
    with open(filename) as file:
        stocks = np.array(file.read().split('\n'))
    picks = [random.randrange(505) for i in range(n)]
    return stocks[picks]


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 10, X[:, 0].max() + 10
    x2_min, x2_max = X[:, 1].min() - 10, X[:, 1].max() + 10
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx])

    X_test, y_test = X[test_idx, :], y[test_idx]

    plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidths=1, marker='o', s=55)


print(random_stocks())
print("['ECL' 'AEP' 'AVY' 'CELG' 'DGX' 'AYI' 'MCK' 'ITW' 'AIV' 'XL' "
      "'AON' 'PNR' 'MTD' 'TWX' 'LNT' 'DTE' 'MKC' 'TXN' 'SO' 'DHI']")
np.random.seed(0)
X_xor = np.array([[1.944, 0.9995], [1.184, 0.7647], [1.784, 0.9793], [1.402, 3.614], [0.2156, 1.57], [1.284, 1.967], [1.844, 1.116], [2.995, 1.774], [0.371, 0.5136], [0.0174, 3.292],
                  [0.7359, 1.097], [0.152, 2.133], [2.939, 1.524], [0.6047, 1.759], [8.412, 0.6995], [0.5369, 1.318], [1.486, 1.094], [1.322, 3.309], [11.76, 0.914], [0.7796, 6.639]])
y_xor = np.array([-1, 1, 1, -1, -1, -1, -1, 1, -1, -1,
                  1, 1, 1, -1, 1, 1, -1, 1, -1, 1])

plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            c='b', marker='x',
            label='Beat the S&P 500')
plt.scatter(X_xor[y_xor == -1, 0],
            X_xor[y_xor == -1, 1],
            c='r',
            marker='s',
            label='Lost to the S&P 500')

plt.xlabel('PEG Ratio')
plt.ylabel('Current Ratio')
plt.xlim(-1, 12)
plt.ylim(0, 7)
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Create a SVC classifier using an RBF kernel
svm = SVC()
# Train the classifier
svm.fit(X_xor, y_xor)
print("['ROST' 'ALL' 'CBS' 'TGT' 'GILD' 'FCX' 'ES' 'C' 'NDAQ' 'BF.B']")
print('[-1 1 -1 -1 -1 -1 1 1 1 1]')
print(svm.predict([[2.303, 1.523], [0.613, 8.620], [1.28, 1.762], [2.787, 1.016], [0.3569, 1.935], [-0.014, 2.564], [5.643, 0.9139], [3.217, 4.980], [0.6811, 1.072], [0.2951, 2.799]]))


# Visualize the decision boundaries
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.xlabel('PEG Ratio')
plt.ylabel('Current Ratio')
plt.xlim(-1, 12)
plt.ylim(0, 7)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()