#include <OpenANN/RBM.h>
#include <OpenANN/ActivationFunctions.h>
#include <OpenANN/util/EigenWrapper.h>
#include <OpenANN/util/OpenANNException.h>
#include <OpenANN/io/Logger.h>

namespace OpenANN
{

RBM::RBM(int D, int H, int cdN, double stdDev, bool backprop,
         Regularization regularization)
  : D(D), H(H), cdN(cdN), stdDev(stdDev),
    W(H, D), posGradW(H, D), negGradW(H, D), Wd(H, D),
    bv(D), posGradBv(D), negGradBv(D),
    bh(H), posGradBh(H), negGradBh(H), bhd(H),
    pv(1, D), v(1, D), ph(1, H), h(1, H), phd(1, H), K(D* H + D + H),
    deltas(1, H), e(1, D), params(K), grad(K),
    backprop(backprop), regularization(regularization)
{
  initialize();
}

Eigen::VectorXd RBM::operator()(const Eigen::VectorXd& x)
{
  v = x.transpose();
  sampleHgivenV();
  return ph.transpose();
}

Eigen::MatrixXd RBM::operator()(const Eigen::MatrixXd& X)
{
  v = X;
  sampleHgivenV();
  return ph;
}

bool RBM::providesInitialization()
{
  return true;
}

void RBM::initialize()
{
  int idx = 0;
  for(int j = 0; j < H; j++)
    for(int i = 0; i < D; i++)
    {
      W(j, i) = rng.sampleNormalDistribution<double>() * stdDev;
      params(idx++) = W(j, i);
    }
  bv.setZero();
  for(int i = 0; i < D; i++)
    params(idx++) = bv(i);
  bh.setZero();
  for(int j = 0; j < H; j++)
    params(idx++) = bh(j);
  setParameters(params);
}

unsigned int RBM::examples()
{
  return trainSet->samples();
}

unsigned int RBM::dimension()
{
  return K;
}

void RBM::setParameters(const Eigen::VectorXd& parameters)
{
  params = parameters;
  int idx = 0;
  for(int j = 0; j < H; j++)
    for(int i = 0; i < D; i++)
      W(j, i) = parameters(idx++);
  for(int i = 0; i < D; i++)
    bv(i) = parameters(idx++);
  for(int j = 0; j < H; j++)
    bh(j) = parameters(idx++);
  OPENANN_CHECK_MATRIX_BROKEN(parameters);
}

const Eigen::VectorXd& RBM::currentParameters()
{
  return params;
}

double RBM::error()
{
  double e = 0.0;
  for(int n = 0; n < trainSet->samples(); n++)
    e += error(n);
  return e;
}

double RBM::error(unsigned int n)
{
  return (reconstructProb(n, 1) - trainSet->getInstance(n).transpose()).squaredNorm();
}

bool RBM::providesGradient()
{
  return true;
}

Eigen::VectorXd RBM::gradient()
{
  Eigen::VectorXd grad(K);
  grad.setZero();
  for(int n = 0; n < trainSet->samples(); n++)
    grad += gradient(n);
  return grad;
}

Eigen::VectorXd RBM::gradient(unsigned int n)
{
  v = trainSet->getInstance(n).transpose();
  reality();
  daydream();
  fillGradient();
  return grad;
}

void RBM::errorGradient(std::vector<int>::const_iterator startN,
                        std::vector<int>::const_iterator endN,
                        double& value, Eigen::VectorXd& grad)
{
  const int N = endN - startN;
  v.conservativeResize(N, trainSet->inputs());
  int n = 0;
  for(std::vector<int>::const_iterator it = startN; it != endN; ++it, ++n)
    v.row(n) = trainSet->getInstance(*it);
  reality();
  daydream();
  fillGradient();
  grad = this->grad;
  n = 0;
  value = 0.0;
  for(std::vector<int>::const_iterator it = startN; it != endN; ++it, ++n)
    value += (trainSet->getInstance(*it) - pv.row(n).transpose()).squaredNorm();
}

OutputInfo RBM::initialize(std::vector<double*>& parameterPointers,
                           std::vector<double*>& parameterDerivativePointers)
{
  if(backprop)
  {
    for(int j = 0; j < H; j++)
    {
      for(int i = 0; i < D; i++)
      {
        parameterPointers.push_back(&W(j, i));
        parameterDerivativePointers.push_back(&Wd(j, i));
      }
    }
    for(int j = 0; j < H; j++)
    {
      parameterPointers.push_back(&bh(j));
      parameterDerivativePointers.push_back(&bhd(j));
    }
  }

  OutputInfo info;
  info.dimensions.push_back(H);
  return info;
}

Eigen::VectorXd RBM::getParameters()
{
  return currentParameters();
}

void RBM::forwardPropagate(Eigen::MatrixXd* x, Eigen::MatrixXd*& y, bool dropout)
{
  v = *x;
  sampleHgivenV();
  y = &ph;
}

void RBM::backpropagate(Eigen::MatrixXd* ein, Eigen::MatrixXd*& eout,
                        bool backpropToPrevious)
{
  const int N = ph.rows();
  phd.conservativeResize(N, Eigen::NoChange);
  // Derive activations
  activationFunctionDerivative(LOGISTIC, ph, phd);
  deltas = phd.cwiseProduct(*ein);
  if(backprop)
  {
    Wd = deltas.transpose() * v;
    bhd = deltas.colwise().sum().transpose();
  }
  // Prepare error signals for previous layer
  if(backpropToPrevious)
    e = deltas * W;
  eout = &e;
}

Eigen::MatrixXd& RBM::getOutput()
{
  return ph;
}

int RBM::visibleUnits()
{
  return D;
}

int RBM::hiddenUnits()
{
  return H;
}

const Eigen::MatrixXd& RBM::getWeights()
{
  return W;
}

const Eigen::MatrixXd& RBM::getVisibleProbs()
{
  return pv;
}

const Eigen::MatrixXd& RBM::getVisibleSample()
{
  return v;
}

Eigen::MatrixXd RBM::reconstructProb(int n, int steps)
{
  v = trainSet->getInstance(n).transpose();
  pv = v;
  for(int i = 0; i < steps; i++)
  {
    sampleHgivenV();
    sampleVgivenH();
  }
  return pv;
}

void RBM::sampleHgivenV()
{
  const int N = v.rows();
  h.conservativeResize(N, Eigen::NoChange);
  ph = v * W.transpose();
  ph.rowwise() += bh.transpose();
  activationFunction(LOGISTIC, ph, ph);
  for(int n = 0; n < N; n++)
    for(int j = 0; j < H; j++)
      h(n, j) = (double)(ph(n, j) > rng.generate<double>(0.0, 1.0));
}

void RBM::sampleVgivenH()
{
  const int N = h.rows();
  pv = h * W;
  pv.rowwise() += bv.transpose();
  activationFunction(LOGISTIC, pv, pv);
  for(int n = 0; n < N; n++)
    for(int i = 0; i < D; i++)
      v(n, i) = (double)(pv(n, i) > rng.generate<double>(0.0, 1.0));
}

void RBM::reality()
{
  sampleHgivenV();

  posGradW = ph.transpose() * v;
  posGradBv = v.colwise().sum().transpose();
  posGradBh = ph.colwise().sum().transpose();
}

void RBM::daydream()
{
  for(int n = 0; n < cdN; n++)
  {
    sampleVgivenH();
    sampleHgivenV();
  }

  negGradW = ph.transpose() * pv;
  negGradBv = pv.colwise().sum().transpose();
  negGradBh = ph.colwise().sum().transpose();
}

void RBM::fillGradient()
{
  int idx = 0;
  for(int j = 0; j < H; j++)
    for(int i = 0; i < D; i++)
      grad(idx++) = posGradW(j, i) - negGradW(j, i);
  for(int i = 0; i < D; i++)
    grad(idx++) = posGradBv(i) - negGradBv(i);
  for(int j = 0; j < H; j++)
    grad(idx++) = posGradBh(j) - negGradBh(j);
  if(regularization.l1Penalty > 0.0)
  {
    idx = 0;
    for(int j = 0; j < H; j++)
      for(int i = 0; i < D; i++)
        grad(idx++) -= regularization.l1Penalty * W(j, i) / std::abs(W(j, i));
  }
  if(regularization.l2Penalty > 0.0)
  {
    idx = 0;
    for(int j = 0; j < H; j++)
      for(int i = 0; i < D; i++)
        grad(idx++) -= regularization.l2Penalty * W(j, i);
  }
  grad *= -1.0;
}

}
