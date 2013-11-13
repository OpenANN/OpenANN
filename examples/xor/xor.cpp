#include <OpenANN/OpenANN>
#include <OpenANN/io/Logger.h>
#include <OpenANN/io/DirectStorageDataSet.h>
#include <OpenANN/util/Random.h>
#include <Eigen/Core>
#include <iostream>

using namespace OpenANN;

int main()
{
  // Create dataset
  const int D = 2; // number of inputs
  const int F = 1; // number of outputs
  const int N = 4; // size of training set
  Eigen::MatrixXd X(N, D); // inputs
  Eigen::MatrixXd T(N, F); // desired outputs (targets)
  // Each row represents an instance
  X.row(0) << 0.0, 1.0;
  T.row(0) << 1.0;
  X.row(1) << 0.0, 0.0;
  T.row(1) << 0.0;
  X.row(2) << 1.0, 1.0;
  T.row(2) << 0.0;
  X.row(3) << 1.0, 0.0;
  T.row(3) << 1.0;
  DirectStorageDataSet dataSet(&X, &T);

  // Make the result repeatable
  RandomNumberGenerator().seed(0);

  // Create network
  Net net;
  // Add an input layer with D inputs, 1 hidden layer with 2 nodes and an
  // output layer with F outputs. Use logistic activation function in hidden
  // layer and output layer.
  makeMLNN(net, LOGISTIC, LOGISTIC, D, F, 1, 2);
  // Add training set
  net.trainingSet(dataSet);

  // Set stopping conditions
  StoppingCriteria stop;
  stop.minimalValueDifferences = 1e-10;
  // Train network, i.e. minimize sum of squared errors (SSE) with
  // Levenberg-Marquardt optimization algorithm until the stopping criteria
  // are satisfied.
  train(net, "LMA", MSE, stop);

  // Use network to predict labels of the training data
  for(int n = 0; n < N; n++)
  {
    Eigen::VectorXd y = net(dataSet.getInstance(n));
    std::cout << y << std::endl;
  }

  return 0;
}
