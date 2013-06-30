#include <OpenANN/SparseAutoEncoder.h>
#include <OpenANN/layers/FullyConnected.h>
#include <OpenANN/util/AssertionMacros.h>
#include <OpenANN/util/Random.h>

namespace OpenANN
{

SparseAutoEncoder::SparseAutoEncoder(int D, int H, double beta, double rho,
                                     double lambda, ActivationFunction act)
  : D(D), H(H), beta(beta), rho(rho), lambda(lambda), act(act), W1(H, D),
    W2(D, H), b1(H), b2(D)
{
  initialize();
}

Eigen::VectorXd SparseAutoEncoder::operator()(const Eigen::VectorXd& x)
{
  A1 = x.transpose() * W1.transpose();
  Z1.conservativeResize(A1.rows(), Eigen::NoChange);
  activationFunction(act, A1, Z1);
  return Z2.transpose();
}

Eigen::MatrixXd SparseAutoEncoder::operator()(const Eigen::MatrixXd& X)
{
  A1 = X * W1.transpose();
  Z1.conservativeResize(A1.rows(), Eigen::NoChange);
  activationFunction(act, A1, Z1);
  return Z2;
}

bool SparseAutoEncoder::providesInitialization()
{
  return true;
}

void SparseAutoEncoder::initialize()
{
  RandomNumberGenerator rng;
  double r = std::sqrt(6.0) / std::sqrt(H + D + 1.0);
  for(int j = 0; j < H; j++)
  {
    for(int i = 0; i < D; i++)
    {
      W1(j, i) = rng.generate<double>(-r, 2*r);
      W2(i, j) = rng.generate<double>(-r, 2*r);
    }
  }
  b1.setZero();
  b2.setZero();
  pack(parameters, W1, W2, b1, b2);
}

unsigned int SparseAutoEncoder::dimension()
{
  return 2*D*H + D + H;
}

void SparseAutoEncoder::setParameters(const Eigen::VectorXd& parameters)
{
  this->parameters = parameters;
  unpack(parameters, W1, W2, b1, b2);
}

const Eigen::VectorXd& SparseAutoEncoder::currentParameters()
{
  return parameters;
}

double SparseAutoEncoder::error()
{
  // TODO
}

bool SparseAutoEncoder::providesGradient()
{
  return true;
}

Eigen::VectorXd SparseAutoEncoder::gradient()
{
  // TODO forward
  const int N = a.rows();
  yd.conservativeResize(N, Eigen::NoChange);
  // Derive activations
  activationFunctionDerivative(act, y, yd);
  deltas = yd.cwiseProduct(*ein);
  deltas.array().rowwise() += beta *
      ((1.0 - rho) * (1.0 - meanActivation->array()).inverse() -
      rho * meanActivation->array().inverse());
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
  // TODO pack
}

void SparseAutoEncoder::errorGradient(double& value, Eigen::VectorXd& grad)
{
  Eigen::VectorXd meanActivation = Eigen::VectorXd(H);
  meanActivation.setZero();
  for(int n = 0; n < trainSet->samples(); n++)
    meanActivation += (*this)(trainSet->getInstance(n));
  meanActivation /= trainSet->samples();
  // TODO
  value += beta * (rho * (rho * meanActivation.array().inverse()).log() +
      (1 - rho) * ((1 - rho) * (1 - meanActivation.array()).inverse()).log()).sum();
}

Eigen::MatrixXd SparseAutoEncoder::getInputWeights()
{
  return W1;
}

Eigen::MatrixXd SparseAutoEncoder::getOutputWeights()
{
  return W2;
}

Eigen::VectorXd SparseAutoEncoder::reconstruct(const Eigen::VectorXd& x)
{
  A1 = x.transpose() * W1.transpose();
  Z1.conservativeResize(A1.rows(), Eigen::NoChange);
  activationFunction(act, A1, Z1);
  A2 = Z1 * W2.transpose();
  Z2.conservativeResize(A2.rows(), Eigen::NoChange);
  activationFunction(act, A2, Z2);
  return Z2.transpose();
}

void SparseAutoEncoder::pack(Eigen::VectorXd& vector,
                             const Eigen::MatrixXd& W1,
                             const Eigen::MatrixXd& W2,
                             const Eigen::VectorXd& b1,
                             const Eigen::VectorXd& b2)
{
  int idx = 0;
  for(int j = 0; j < H; j++)
  {
    for(int i = 0; i < D; i++)
    {
      vector(idx++) = W1(j, i);
      vector(idx++) = W2(i, j);
    }
  }
  for(int j = 0; j < H; j++)
    vector(idx++) = b1(j);
  for(int i = 0; i < D; i++)
    vector(idx++) = b2(i);
}

void SparseAutoEncoder::unpack(const Eigen::VectorXd& vector,
                               Eigen::MatrixXd& W1, Eigen::MatrixXd& W2,
                               Eigen::VectorXd& b1, Eigen::VectorXd& b2)
{
  int idx = 0;
  for(int j = 0; j < H; j++)
  {
    for(int i = 0; i < D; i++)
    {
      W1(j, i) = vector(idx++);
      W2(i, j) = vector(idx++);
    }
  }
  for(int j = 0; j < H; j++)
    b1(j) = vector(idx++);
  for(int i = 0; i < D; i++)
    b2(i) = vector(idx++);
}

} // namespace OpenANN
