#include <OpenANN/SparseAutoEncoder.h>

namespace OpenANN
{

SparseAutoEncoder::SparseAutoEncoder(int D, int H, ActivationFunction act,
                                     double beta, double rho)
  : D(D), H(H), beta(beta), rho(rho)
{
  setRegularization(0.0, 0.0, 0.0, rho, beta);
  inputLayer(D);
  fullyConnectedLayer(H, act);
  setRegularization(0.0, 0.0, 0.0, 0.0, 0.0);
  outputLayer(D, LINEAR);
}

Eigen::VectorXd SparseAutoEncoder::operator()(const Eigen::VectorXd& x)
{
  tempInput = x.transpose();
  Eigen::MatrixXd* y = &tempInput;
  layers[0]->forwardPropagate(y, y, dropout);
  tempOutput = *y;
  return tempOutput.transpose();
}

Eigen::MatrixXd SparseAutoEncoder::operator()(const Eigen::MatrixXd& X)
{
  tempInput = X;
  Eigen::MatrixXd* y = &tempInput;
  layers[0]->forwardPropagate(y, y, dropout);
  tempOutput = *y;
  return tempOutput;
}

void SparseAutoEncoder::errorGradient(double& value, Eigen::VectorXd& grad)
{
  /*Eigen::VectorXd meanActivation = Eigen::VectorXd(H);
  meanActivation.setZero();
  for(int n = 0; n < Net::trainSet->samples(); n++)
    meanActivation += (*this)(Net::trainSet->getInstance(n));*/
  Net::errorGradient(value, grad);
}

} // namespace OpenANN
