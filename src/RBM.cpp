#include <OpenANN/RBM.h>
#include <OpenANN/ActivationFunctions.h>
#include <OpenANN/util/EigenWrapper.h>
#include <OpenANN/util/OpenANNException.h>
#include <OpenANN/io/Logger.h>

namespace OpenANN {

RBM::RBM(int D, int H, int cdN, double stdDev, bool backprop)
  : D(D), H(H), cdN(cdN), stdDev(stdDev),
    W(H, D), posGradW(H, D), negGradW(H, D), Wd(H, D),
    bv(D), posGradBv(D), negGradBv(D),
    bh(H), posGradBh(H), negGradBh(H), bhd(H),
    pv(D), v(D), ph(H), h(H), phd(H), K(D*H + D + H), params(K),
    deltas(H), e(D), backprop(backprop)
{
  initialize();
}

Eigen::VectorXd RBM::operator()(const Eigen::VectorXd& x)
{
  v = x;
  sampleHgivenV();
  return ph;
}

bool RBM::providesInitialization()
{
  return true;
}

void RBM::initialize()
{
  for(int k = 0; k < K; k++)
    params(k) = rng.sampleNormalDistribution<double>() * stdDev;
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

Eigen::VectorXd RBM::currentParameters()
{
  return params;
}

double RBM::error()
{
  double error = 0.0;
  for(int n = 0; n < trainSet->samples(); n++)
    error += (reconstructProb(n, 1) - trainSet->getInstance(n)).squaredNorm();
  return error;
}

bool RBM::providesGradient()
{
  return true;
}

Eigen::VectorXd RBM::gradient()
{
  Eigen::VectorXd grad(dimension());
  grad.fill(0.0);
  for(int n = 0; n < trainSet->samples(); n++)
    grad += gradient(n);
  return grad;
}

Eigen::VectorXd RBM::gradient(unsigned int i)
{
  reality(i);
  daydream();

  Eigen::VectorXd gradient(dimension());
  int idx = 0;
  for(int j = 0; j < H; j++)
    for(int i = 0; i < D; i++)
      gradient(idx++) = posGradW(j, i) - negGradW(j, i);
  for(int i = 0; i < D; i++)
    gradient(idx++) = posGradBv(i) - negGradBv(i);
  for(int j = 0; j < H; j++)
    gradient(idx++) = posGradBh(j) - negGradBh(j);
  return -gradient;
}

bool RBM::providesHessian()
{
  return false;
}

Eigen::MatrixXd RBM::hessian()
{
  throw OpenANNException("RBM::hessian() is not implemented!");
}

Learner& RBM::trainingSet(Eigen::MatrixXd& trainingInput,
                          Eigen::MatrixXd& trainingOutput)
{
  throw OpenANNException("RBM::trainingSet(input, output) is not implemented!");
}

Learner& RBM::trainingSet(DataSet& trainingSet)
{
  trainSet = &trainingSet;
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
      parameterDerivativePointers.push_back(&bh(j));
    }
  }

  OutputInfo info;
  info.dimensions.push_back(H);
  return info;
}

void RBM::backpropagate(Eigen::VectorXd* ein, Eigen::VectorXd*& eout)
{
  // Derive activations
  activationFunctionDerivative(LOGISTIC, ph, phd);
  for(int j = 0; j < H; j++)
    deltas(j) = phd(j) * (*ein)(j);
  // Weight derivatives
  if(backprop)
    Wd = deltas * v.transpose();
  // Prepare error signals for previous layer
  e = W.transpose() * deltas;
  eout = &e;
}

void RBM::forwardPropagate(Eigen::VectorXd* x, Eigen::VectorXd*& y, bool dropout)
{
  v = *x;
  sampleHgivenV();
  y = &ph;
}

Eigen::VectorXd& RBM::getOutput()
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

const Eigen::VectorXd& RBM::getVisibleProbs()
{
  return pv;
}

const Eigen::VectorXd& RBM::getVisibleSample()
{
  return v;
}

Eigen::VectorXd RBM::reconstructProb(int n, int steps)
{
  v = trainSet->getInstance(n);
  pv = v;
  for(int i = 0; i < steps; i++)
  {
    sampleHgivenV();
    sampleVgivenH();
  }
  return pv;
}

Eigen::VectorXd RBM::reconstruct(int n, int steps)
{
  v = trainSet->getInstance(n);
  for(int i = 0; i < steps; i++)
  {
    sampleHgivenV();
    sampleVgivenH();
  }
  return v;
}

void RBM::reality(int n)
{
  v = trainSet->getInstance(n);

  sampleHgivenV();

  posGradW = ph * v.transpose();
  posGradBv = v;
  posGradBh = ph;
}

void RBM::daydream()
{
  for(int n = 0; n < cdN; n++)
  {
    sampleVgivenH();
    sampleHgivenV();
  }

  negGradW = ph * pv.transpose();
  negGradBv = pv;
  negGradBh = ph;
}

void RBM::sampleHgivenV()
{
  ph = W * v + bh;
  activationFunction(LOGISTIC, ph, ph);
  for(int j = 0; j < H; j++)
    h(j) = (double) (ph(j) > rng.generate<double>(0.0, 1.0));
}

void RBM::sampleVgivenH()
{
  pv = W.transpose() * h + bv;
  activationFunction(LOGISTIC, pv, pv);
  for(int i = 0; i < D; i++)
    v(i) = (double) (pv(i) > rng.generate<double>(0.0, 1.0));
}

}
