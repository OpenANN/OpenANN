#include <OpenANN/SparseAutoEncoder.h>
#include <OpenANN/layers/FullyConnected.h>
#include <OpenANN/util/AssertionMacros.h>

namespace OpenANN
{

class SparseFullyConnected : public FullyConnected
{
public:
  double beta, rho;
  Eigen::VectorXd* meanActivation;

  SparseFullyConnected(OutputInfo info, int J, bool bias,
                       ActivationFunction act, double stdDev,
                       Regularization regularization, double beta, double rho)
    : FullyConnected(info, J, bias, act, stdDev, regularization), beta(beta),
      rho(rho), meanActivation(0)
  {}

  virtual void backpropagate(Eigen::MatrixXd* ein, Eigen::MatrixXd*& eout)
  {
    OPENANN_CHECK(meanActivation);
    const int N = a.rows();
    yd.conservativeResize(N, Eigen::NoChange);
    // Derive activations
    activationFunctionDerivative(act, y, yd);
    deltas = yd.cwiseProduct(*ein);
    deltas.array().rowwise() -= beta *
        (-rho * meanActivation->array().inverse() +
        (1.0 - rho) * (1.0 - meanActivation->array()).inverse());
    // Weight derivatives
    Wd = deltas.transpose() * *x;
    if(bias)
      bd = deltas.colwise().sum().transpose();
    if(regularization.l1Penalty > 0.0)
      Wd.array() += regularization.l1Penalty * W.array() / W.array().abs();
    if(regularization.l2Penalty > 0.0)
      Wd += regularization.l2Penalty * W;
    // Prepare error signals for previous layer
    e = deltas * W;
    eout = &e;
  }
};

SparseAutoEncoder::SparseAutoEncoder(int D, int H, ActivationFunction act,
                                     double beta, double rho)
  : D(D), H(H), beta(beta), rho(rho)
{
  inputLayer(D);
  addLayer(new FullyConnected(infos.back(), H, true, act, 0.05, // TODO stdDev as parameter
                              regularization));
  fullyConnectedLayer(H, act);
  outputLayer(D, LINEAR);
}

Eigen::VectorXd SparseAutoEncoder::operator()(const Eigen::VectorXd& x)
{
  tempInput = x.transpose();
  Eigen::MatrixXd* y = &tempInput;
  layers[0]->forwardPropagate(y, y, dropout);
  layers[1]->forwardPropagate(y, y, dropout);
  tempOutput = *y;
  return tempOutput.transpose();
}

Eigen::MatrixXd SparseAutoEncoder::operator()(const Eigen::MatrixXd& X)
{
  tempInput = X;
  Eigen::MatrixXd* y = &tempInput;
  layers[0]->forwardPropagate(y, y, dropout);
  layers[1]->forwardPropagate(y, y, dropout);
  tempOutput = *y;
  return tempOutput;
}

void SparseAutoEncoder::errorGradient(double& value, Eigen::VectorXd& grad)
{
  Eigen::VectorXd meanActivation = Eigen::VectorXd(H);
  meanActivation.setZero();
  for(int n = 0; n < Net::trainSet->samples(); n++)
    meanActivation += (*this)(Net::trainSet->getInstance(n));
  ((SparseFullyConnected&) getLayer(1)).meanActivation = &meanActivation;
  Net::errorGradient(value, grad);
  ((SparseFullyConnected&) getLayer(1)).meanActivation = 0;
}

Eigen::MatrixXd SparseAutoEncoder::getInputWeights()
{
  return ((FullyConnected*) layers[1])->getWeights();
}

Eigen::MatrixXd SparseAutoEncoder::getOutputWeights()
{
  return ((FullyConnected*) layers[2])->getWeights();
}

Eigen::VectorXd SparseAutoEncoder::reconstruct(const Eigen::VectorXd& x)
{
  tempInput = x.transpose();
  forwardPropagate();
  return tempOutput.transpose();
}

} // namespace OpenANN
