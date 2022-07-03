from preprocessing import X_train, X_test, y_train_, y_test_
from sklearn import svm


train_n_samples, dim1, dim2, dim3 = X_train.shape
test_n_samples, _, _, _ = X_test.shape
X_train = X_train.reshape((train_n_samples, dim1 * dim2 * dim3))
X_test = X_test.reshape((test_n_samples, dim1 * dim2 * dim3))

# select different type of kernel function and compare the score

# kernel = 'rbf'
clf_rbf = svm.SVC(kernel='rbf')
clf_rbf.fit(X_train, y_train_)
score_rbf = clf_rbf.score(X_test, y_test_)
print("The score of rbf is : %f" % score_rbf)

# kernel = 'linear'
clf_linear = svm.SVC(kernel='linear')
clf_linear.fit(X_train, y_train_)
score_linear = clf_linear.score(X_test, y_test_)
print("The score of linear is : %f" % score_linear)

# kernel = 'poly'
clf_poly = svm.SVC(kernel='poly')
clf_poly.fit(X_train, y_train_)
score_poly = clf_poly.score(X_test, y_test_)
print("The score of poly is : %f" % score_poly)
