import pylab
import scipy.io
from openann import *

a = scipy.io.loadmat("sarcos_inv.mat")
X = a["sarcos_inv"][:, :21]
Y = a["sarcos_inv"][:, 21:]
b = scipy.io.loadmat("sarcos_inv_test.mat")
Xtest = b["sarcos_inv_test"][:, :21]
Ytest = b["sarcos_inv_test"][:, 21:]

# Normalize data
Xmu = X.mean(axis=0)
Xstd = X.std(axis=0)
X = (X - Xmu) / Xstd
Xtest = (Xtest - Xmu) / Xstd
Ymu = Y.mean(axis=0)
Ystd = Y.std(axis=0)
Y = (Y - Ymu) / Ystd
Ytest = (Ytest - Ymu) / Ystd
# Learn only one dimension
#Y = Y[:, 0, numpy.newaxis]
#Ytest = Ytest[:, 0, numpy.newaxis]

training_set = Dataset(X, Y)
validation_set = Dataset(Xtest, Ytest)

D = X.shape[1]
F = Y.shape[1]
net = Net()
net.input_layer(D)
net.fully_connected_layer(250, Activation.TANH)
net.fully_connected_layer(100, Activation.TANH)
net.output_layer(F, Activation.LINEAR)

stop_dict = {"maximal_iterations" : 100}
opt = MBSGD(stop_dict, learning_rate=0.2, learning_rate_decay=0.9999, min_learning_rate=0.001, momentum=0.5, batch_size=128)
opt.optimize(net, training_set)

print "RMSE"
print "Training:", rmse(net, training_set)
print "Test:", rmse(net, validation_set)

pred = net.predict(X)
pred_test = net.predict(Xtest)
var = Y.var(axis=0) # in case we do not normalize the outputs
for f in range(F):
    nMSE_train = sum((Y[:, f] - pred[:, f])**2) / len(Y) / var[f]
    nMSE_test = sum((Ytest[:, f] - pred_test[:, f])**2) / len(Ytest) / var[f]
    print "Dimension %d: nMSE = %f%% / %f%%" % (f, nMSE_train*100, nMSE_test*100)

dim = 0
n_samples = 200
pylab.plot(Ytest[:n_samples, dim], label="Actual")
pylab.plot(net.predict(Xtest[:n_samples])[:, dim], label="Predicted")
pylab.legend(loc="best")
pylab.title("Comparison of %d samples from dimension %d (test set)" % (n_samples, dim+1))
pylab.show()
