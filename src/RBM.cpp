#include <OpenANN/RBM.h>
#include <OpenANN/ActivationFunctions.h>
#include <OpenANN/util/EigenWrapper.h>
#include <OpenANN/io/Logger.h>

namespace OpenANN {

RBM::RBM(int D, int H, int cdN, double stdDev)
  : D(D), H(H), cdN(cdN), stdDev(stdDev),
    W(H, D), posGradW(H, D), negGradW(H, D),
    bv(D), posGradBv(D), negGradBv(D),
    bh(H), posGradBh(H), negGradBh(H),
    pv(D), v(D), ph(H), h(H), K(D*H + D + H), params(K)
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
  return 0.0; // TODO reconstruction error?
}

bool RBM::providesGradient()
{
  return true;
}

Eigen::VectorXd RBM::gradient()
{
  // TODO CD-n
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
  // TODO return dummy
}

Learner& RBM::trainingSet(Eigen::MatrixXd& trainingInput,
                          Eigen::MatrixXd& trainingOutput)
{
  // TODO
}

Learner& RBM::trainingSet(DataSet& trainingSet)
{
  trainSet = &trainingSet;
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
