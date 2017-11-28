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
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

    X_test, y_test = X[test_idx, :], y[test_idx]

    plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidths=1, marker='o', s=55)


print(random_stocks())
print("['ECL' 'AEP' 'AVY' 'CELG' 'DGX' 'AYI' 'MCK' 'ITW' 'AIV' 'XL' "
      "'AON' 'PNR' 'MTD' 'TWX' 'LNT' 'DTE' 'MKC' 'TXN' 'NDAQ' 'DHI']")
np.random.seed(0)
X_xor = np.array([[2.244, 1.445], [0.2887, 0.684], [1.087, 1.137], [0.4234, 3.986], [0.2307, 1.375], [1.916, 2.135], [0.3194, 1.113], [0.5269, 2.613], [-0.9164, 0.7491], [0.8687, 6.473],
                  [3.836, 1.125], [1.376, 1.754], [1.602, 1.510], [1.793, 1.430], [3.227, 1.088], [0.7805, 1.182], [5.668, 1.351], [14.43, 2.506], [1.056, 1.651], [0.0069, 5.416]])
y_xor = np.array([-1, -1, 1, 1, -1, 1, -1, 1, -1, -1,
                  1, -1, 1, 1, 1, 1, -1, 1, 1, 1])

plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            c='b', marker='x',
            label='Beat the S&P 500')
plt.scatter(X_xor[y_xor == -1, 0],
            X_xor[y_xor == -1, 1],
            c='r',
            marker='s',
            label='Lost to the S&P 500')

plt.xlim(-1, 15)
plt.ylim(-1, 15)
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Create a SVC classifier using an RBF kernel
svm = SVC()
# Train the classifier
svm.fit(X_xor, y_xor)
print(svm.predict([[0.7478, 1.467]]))

# Visualize the decision boundaries
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.xlim(-1, 15)
plt.ylim(-1, 15)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()