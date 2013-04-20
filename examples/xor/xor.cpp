#include <OpenANN/OpenANN>
#include <Eigen/Dense>
#include <iostream>

using namespace OpenANN;

int main()
{
  // Create dataset
  const int D = 2;
  const int F = 1;
  const int N = 4;
  Eigen::MatrixXd x(D, N);
  Eigen::MatrixXd t(F, N);
  x.col(0) << 0.0, 1.0; t.col(0) << 1.0;
  x.col(1) << 0.0, 0.0; t.col(1) << 0.0;
  x.col(2) << 1.0, 1.0; t.col(2) << 0.0;
  x.col(3) << 1.0, 0.0; t.col(3) << 1.0;

  // Create network
  Net net;
  net.inputLayer(D)
     .fullyConnectedLayer(3, TANH)
     .outputLayer(F, TANH)
     .trainingSet(x, t);

  // Train network
  StoppingCriteria stop;
  stop.minimalValueDifferences = 1e-10;
  train(net, "LMA", SSE, stop);

  // Use network
  for(int n = 0; n < N; n++)
  {
    Eigen::VectorXd y = net(x.col(n));
    std::cout << y << std::endl;
  }

  return 0;
}
