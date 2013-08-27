#include <OpenANN/SparseAutoEncoder.h>
#include <OpenANN/layers/FullyConnected.h>
#include <OpenANN/util/AssertionMacros.h>
#include <OpenANN/util/Random.h>
#include <OpenANN/util/EigenWrapper.h>

namespace OpenANN
{

SparseAutoEncoder::SparseAutoEncoder(int D, int H, double beta, double rho,
                                     double lambda, ActivationFunction act)
  : D(D), H(H), beta(beta), rho(rho), lambda(lambda), act(act), W1(H, D),
    W2(D, H), W1d(H, D), W2d(D, H), b1(H), b2(D), b1d(H), b2d(D)
{
  parameters.resize(dimension());
  grad.resize(dimension());
  initialize();
}

Eigen::VectorXd SparseAutoEncoder::operator()(const Eigen::VectorXd& x)
{
  Eigen::MatrixXd X = x.transpose();
  return (*this)(X).transpose();
}

Eigen::MatrixXd SparseAutoEncoder::operator()(const Eigen::MatrixXd& X)
{
  A1 = X * W1.transpose();
  A1.rowwise() += b1.transpose();
  Z1.conservativeResize(A1.rows(), A1.cols());
  activationFunction(act, A1, Z1);
  return Z1;
}

bool SparseAutoEncoder::providesInitialization()
{
  return true;
}

void SparseAutoEncoder::initialize()
{
  initializeParameters();
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
  const int N = X.rows();
  (*this)(X);
  A2 = Z1 * W2.transpose();
  A2.rowwise() += b2.transpose();
  Z2.conservativeResize(A2.rows(), A2.cols());
  activationFunction(act, A2, Z2);

  meanActivation = Z1.colwise().sum().transpose() / N;

  dEdZ2 = Z2 - X;
  double err = dEdZ2.array().square().sum() / (2.0*N);
  err += lambda/2.0 * (W1.array().square().sum() + W2.array().square().sum());
  // KL divergence to target distribution:
  // beta * sum[(rho * log(rho/rho_hat)) + (1-rho) * log((1-rho)/(1-rho_hat))]
  err += beta * (rho * (rho * meanActivation.array().inverse()).log() +
      (1-rho) * ((1-rho) * (1-meanActivation.array()).inverse()).log()).sum();
  return err;
}

bool SparseAutoEncoder::providesGradient()
{
  return true;
}

Eigen::VectorXd SparseAutoEncoder::gradient()
{
  double error;
  errorGradient(error, grad);
  return grad;
}

void SparseAutoEncoder::errorGradient(double& value, Eigen::VectorXd& grad)
{
  const int N = X.rows();
  // Forward propagation and error calculation
  value = error();

  G2D.conservativeResize(Z2.rows(), Z2.cols());
  activationFunctionDerivative(act, Z2, G2D);
  Eigen::MatrixXd deltas2 = dEdZ2.cwiseProduct(G2D);
  W2d = deltas2.transpose() * Z1 / N + lambda * W2;
  b2d = deltas2.colwise().sum().transpose() / N;
  Eigen::MatrixXd dEdZ1 = deltas2 * W2;
  dEdZ1.array().rowwise() += beta *
      (-rho * meanActivation.array().inverse()
       +(1.0-rho) * (1.0-meanActivation.array()).inverse()).transpose();
  G1D.conservativeResize(Z1.rows(), Z1.cols());
  activationFunctionDerivative(act, Z1, G1D);
  Eigen::MatrixXd deltas1 = dEdZ1.cwiseProduct(G1D);
  W1d = deltas1.transpose() * X / N + lambda * W1;
  b1d = deltas1.colwise().sum().transpose() / N;

  pack(grad, W1d, W2d, b1d, b2d);
}

Learner& SparseAutoEncoder::trainingSet(DataSet& trainingSet)
{
  X.conservativeResize(trainingSet.samples(), trainingSet.inputs());
  for(int n = 0; n < trainingSet.samples(); n++)
    X.row(n) = trainingSet.getInstance(n);
  return *this;
}

void SparseAutoEncoder::backpropagate(Eigen::MatrixXd* ein,
                                      Eigen::MatrixXd*& eout,
                                      bool backpropToPrevious)
{
  G1D.conservativeResize(Z1.rows(), Z1.cols());
  activationFunctionDerivative(act, Z1, G1D);
  Eigen::MatrixXd deltas1 = ein->cwiseProduct(G1D);
  W1d = deltas1.transpose() * X + lambda * W1;
  b1d = deltas1.colwise().sum().transpose();
  if(backpropToPrevious)
    dEdZ1 = deltas1 * W1;
  eout = &dEdZ1;
}

void SparseAutoEncoder::forwardPropagate(Eigen::MatrixXd* x,
                                         Eigen::MatrixXd*& y, bool dropout)
{
  X = *x;
  (*this)(*x);
  y = &Z1;
}

Eigen::MatrixXd& SparseAutoEncoder::getOutput()
{
  return Z1;
}

Eigen::VectorXd SparseAutoEncoder::getParameters()
{
  Eigen::VectorXd params(H*D+H);
  int idx = 0;
  for(int h = 0; h < H; h++)
    for(int d = 0; d < D; d++)
      params(idx++) = W1(h, d);
  for(int h = 0; h < H; h++)
    params(idx++) = b1(h);
}

OutputInfo SparseAutoEncoder::initialize(std::vector<double*>& parameterPointers,
                                         std::vector<double*>& parameterDerivativePointers)
{
  for(int h = 0; h < H; h++)
    for(int d = 0; d < D; d++)
    {
      parameterPointers.push_back(&W1(h, d));
      parameterDerivativePointers.push_back(&W1d(h, d));
    }
  for(int h = 0; h < H; h++)
  {
    parameterPointers.push_back(&b1(h));
    parameterDerivativePointers.push_back(&b1d(h));
  }

  OutputInfo info;
  info.dimensions.push_back(H);
  return info;
}

void SparseAutoEncoder::initializeParameters()
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
  (*this)(x);
  A2 = Z1 * W2.transpose();
  A2.rowwise() += b2.transpose();
  Z2.conservativeResize(A2.rows(), A2.cols());
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
