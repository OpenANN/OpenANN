#include <OpenANN/OpenANN>
#include <Eigen/Dense>
#include <OpenANN/io/Logger.h>
#include <iostream>

using namespace OpenANN;

/**
 * \page XOR
 *
 * \section DataSet Data Set
 *
 * The XOR problem cannot be solved by the perceptron (a neural network with
 * just one neuron) and was the reason for the death of neural network
 * research in the 70s until backpropagation was discovered.
 *
 * The data set is simple:
 * <table>
 * <tr>
 * <td>\f$ x_1 \f$</td>
 * <td>\f$ x_2 \f$</td>
 * <td>\f$ y_1 \f$</td>
 * </tr>
 * <tr>
 * <td>0</td>
 * <td>1</td>
 * <td>1</td>
 * </tr>
 * <tr>
 * <td>0</td>
 * <td>0</td>
 * <td>0</td>
 * </tr>
 * <tr>
 * <td>1</td>
 * <td>1</td>
 * <td>0</td>
 * </tr>
 * <tr>
 * <td>1</td>
 * <td>0</td>
 * <td>1</td>
 * </tr>
 * </table>
 *
 * That means \f$ y_1 \f$ is on whenever \f$ x_1 \neq x_2 \f$. The problem is
 * that you cannot draw a line that separates the two classes 0 and 1. They
 * are not linearly separable as you can see in the following picture.
 *
 * \image html xor.png
 *
 * Therefore, we need at least one hidden layer to solve the problem.
 */
int main()
{
  // Create dataset
  const int D = 2;
  const int F = 1;
  const int N = 4;
  Eigen::MatrixXd x(N, D);
  Eigen::MatrixXd t(N, F);
  x.row(0) << 0.0, 1.0; t.row(0) << 1.0;
  x.row(1) << 0.0, 0.0; t.row(1) << 0.0;
  x.row(2) << 1.0, 1.0; t.row(2) << 0.0;
  x.row(3) << 1.0, 0.0; t.row(3) << 1.0;

  // Create network
  Net net;
  net.inputLayer(D)
     .fullyConnectedLayer(2, LOGISTIC)
     .outputLayer(F, LOGISTIC)
     .trainingSet(x, t);

  // Train network
  StoppingCriteria stop;
  stop.minimalValueDifferences = 1e-10;
  train(net, "LMA", SSE, stop);

  // Use network
  for(int n = 0; n < N; n++)
  {
    Eigen::VectorXd y = net(x.row(n));
    std::cout << y << std::endl;
  }

  return 0;
}
