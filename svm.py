from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

plots_directory = ""
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 25

array = []
with open('test1.txt', 'r') as f:
    for line in f:
        array.append(line.split())

new_array = []
for line in array:
    new_line = []
    for num in line:
        new_line.append(float(num))
    new_array.append(new_line)

N = 10
array = np.array(new_array[:N])

points = np.array(array[:,[0,1]])
labels = np.array(array[:,2])

# print('Points: ' + str(points))
# print('Labels: ' + str(labels))

our_linear_svm = svm.SVC(kernel = 'linear', C=100).fit(points, labels)
xy_max = 10

x = np.linspace(-1. * xy_max, 1. * xy_max, N)
y = np.linspace(-1. * xy_max, 1. * xy_max, N)

X,Y = np.meshgrid(x,y)

Z = our_linear_svm.decision_function(np.c_[X.ravel(), Y.ravel()])
Z = Z.reshape(X.shape)

plt.contour(X, Y, Z, [0], colors = 'black', linewidths = 2)
plt.scatter(points[:, 0], points[:, 1], s=100, c=['blue' if label == 1 else 'red' for label in labels])
plt.show()

# print('Support vectors: ' + str(our_linear_svm.support_vectors_))
# print('Dual coefficients: ' + str(our_linear_svm.dual_coef_))
#
# w = sum((our_linear_svm.dual_coef_ * our_linear_svm.support_vectors_.transpose()).transpose())
# print('w vector: ' + str(w))
#
# new_points = np.array([[2, 3], [3, -2], [5, 28], [-3, -4], [-7, 11], [-4, 0]])
# print('New points: ' + str(new_points))
# print('Decision function (distance to separating hyperplane) of new points: ' + str(our_linear_svm.decision_function(new_points)))
# print('Predicted values of new points: ' + str(our_linear_svm.predict(new_points)))
# print()
#
# number_of_points = 100
# number_of_dimensions = 2
#
# points = np.random.random(size = (number_of_points, number_of_dimensions))
# labels = np.where(points[:, 0] > points[:, 1], 1, -1)
#
# # plt.scatter(points[:, 0], points[:, 1], s=100, c=["blue" if label == 1 else "red" for label in labels])
# # plt.show()
#
# our_linear_svm = svm.SVC(kernel = 'linear', C=1.0).fit(points, labels)
# print('Support vectors: ' + str(our_linear_svm.support_vectors_))
# print('Dual coefficients: ' + str(our_linear_svm.dual_coef_))
#
# w = sum((our_linear_svm.dual_coef_ * our_linear_svm.support_vectors_.transpose()).transpose())
# print('w vector: ' + str(w))
# print('b: ' + str(our_linear_svm.intercept_))
#
# new_points = np.array([[0.5, 0.4], [0.4, 0.5], [0.6, 0.3]])
# print('New points: ' + str(new_points))
# print('Decision function (distance to separating hyperplane) of new points: ' + str(our_linear_svm.decision_function(new_points)))
# print('Predicted values of new points: ' + str(our_linear_svm.predict(new_points)))
# print()
#
# points = np.array([[-1, 0], [0, 0], [1, 0]])
# labels = [1, -1, 1]
#
# # plt.scatter(points[:, 0], points[:, 1], s=100, c=["blue" if label == 1 else "red" for label in labels])
# # plt.show()
#
# our_poly_svm_list = [svm.SVC(kernel = 'poly', degree = 2, C = C) for C in [0.01, 0.99, 1., 1.01, 10, 1e2, 1e6]]
# [o.fit(points, labels) for o in our_poly_svm_list]
# print([o.predict(points) for o in our_poly_svm_list])