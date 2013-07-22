#include <OpenANN/OpenANN>
#include <Eigen/Dense>

int main()
{
  int channels = 1, rows = 20, cols = 20;
  int N = 1000;
  int D = channels * rows * cols;
  int F = 2;
  Eigen::MatrixXd X = Eigen::MatrixXd::Random(N, D);
  Eigen::MatrixXd T = Eigen::MatrixXd::Random(N, F);

  OpenANN::Net net;
  net.inputLayer(channels, rows, cols)
  .convolutionalLayer(5, 5, 5, OpenANN::RECTIFIER)
  .outputLayer(F, OpenANN::LINEAR)
  .trainingSet(X, T);
  OpenANN::StoppingCriteria stop;
  stop.maximalIterations = 5;
  OpenANN::train(net, "MBSGD", OpenANN::MSE, stop);
}

