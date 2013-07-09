#include <OpenANN/Net.h>
#include <OpenANN/layers/Input.h>
#include <OpenANN/layers/AlphaBetaFilter.h>
#include <OpenANN/layers/FullyConnected.h>
#include <OpenANN/layers/Compressed.h>
#include <OpenANN/layers/Extreme.h>
#include <OpenANN/layers/Convolutional.h>
#include <OpenANN/layers/Subsampling.h>
#include <OpenANN/layers/MaxPooling.h>
#include <OpenANN/layers/LocalResponseNormalization.h>
#include <OpenANN/layers/Dropout.h>
#include <OpenANN/RBM.h>
#include <OpenANN/IntrinsicPlasticity.h>
#include <OpenANN/io/DirectStorageDataSet.h>
#include <OpenANN/optimization/IPOPCMAES.h>
#include <OpenANN/optimization/LMA.h>
#include <OpenANN/optimization/MBSGD.h>

namespace OpenANN
{

Net::Net()
  : errorFunction(MSE), dropout(false), initialized(false), P(-1), L(0)
{
  layers.reserve(3);
  infos.reserve(3);
}

Net::~Net()
{
  for(int i = 0; i < layers.size(); i++)
  {
    delete layers[i];
    layers[i] = 0;
  }
  layers.clear();
}

Net& Net::inputLayer(int dim1, int dim2, int dim3)
{
  return addLayer(new Input(dim1, dim2, dim3));
}

Net& Net::alphaBetaFilterLayer(double deltaT, double stdDev)
{
  return addLayer(new AlphaBetaFilter(infos.back(), deltaT, stdDev));
}

Net& Net::fullyConnectedLayer(int units, ActivationFunction act, double stdDev,
                              bool bias)
{
  return addLayer(new FullyConnected(infos.back(), units, bias, act, stdDev,
                                     regularization));
}

Net& Net::restrictedBoltzmannMachineLayer(int H, int cdN, double stdDev,
                                          bool backprop)
{
  return addLayer(new RBM(infos.back().outputs(), H, cdN, stdDev,
                          backprop, regularization));
}

Net& Net::compressedLayer(int units, int params, ActivationFunction act,
                          const std::string& compression, double stdDev,
                          bool bias)
{
  return addLayer(new Compressed(infos.back(), units, params, bias, act,
                                 compression, stdDev, regularization));
}

Net& Net::extremeLayer(int units, ActivationFunction act, double stdDev,
                       bool bias)
{
  return addLayer(new Extreme(infos.back(), units, bias, act, stdDev));
}

Net& Net::intrinsicPlasticityLayer(double targetMean, double stdDev)
{
  return addLayer(new IntrinsicPlasticity(infos.back().outputs(), targetMean,
                                          stdDev));
}

Net& Net::convolutionalLayer(int featureMaps, int kernelRows, int kernelCols,
                             ActivationFunction act, double stdDev, bool bias)
{
  return addLayer(new Convolutional(infos.back(), featureMaps, kernelRows,
                                    kernelCols, bias, act, stdDev, regularization));
}

Net& Net::subsamplingLayer(int kernelRows, int kernelCols,
                           ActivationFunction act, double stdDev, bool bias)
{
  return addLayer(new Subsampling(infos.back(), kernelRows, kernelCols, bias,
                                  act, stdDev, regularization));
}

Net& Net::maxPoolingLayer(int kernelRows, int kernelCols)
{
  return addLayer(new MaxPooling(infos.back(), kernelRows, kernelCols));
}

Net& Net::localReponseNormalizationLayer(double k, int n, double alpha,
                                         double beta)
{
  return addLayer(new LocalResponseNormalization(infos.back(), k, n, alpha,
                                                 beta));
}

Net& Net::dropoutLayer(double dropoutProbability)
{
  return addLayer(new Dropout(infos.back(), dropoutProbability));
}

Net& Net::addLayer(Layer* layer)
{
  OPENANN_CHECK(layer != 0);

  OutputInfo info = layer->initialize(parameters, derivatives);
  layers.push_back(layer);
  infos.push_back(info);
  L++;
  return *this;
}

Net& Net::addOutputLayer(Layer* layer)
{
  addLayer(layer);
  initializeNetwork();
  return *this;
}


Net& Net::outputLayer(int units, ActivationFunction act, double stdDev, bool bias)
{
  fullyConnectedLayer(units, act, stdDev, bias);
  initializeNetwork();
  return *this;
}

Net& Net::compressedOutputLayer(int units, int params, ActivationFunction act,
                                const std::string& compression, double stdDev,
                                bool bias)
{
  compressedLayer(units, params, act, compression, stdDev, bias);
  initializeNetwork();
  return *this;
}

unsigned int Net::numberOflayers()
{
  return L;
}

Layer& Net::getLayer(unsigned int l)
{
  OPENANN_CHECK(l >= 0 && l < L);
  return *layers[l];
}

OutputInfo Net::getOutputInfo(unsigned int l)
{
  OPENANN_CHECK(l >= 0 && l < L);
  return infos[l];
}

DataSet* Net::propagateDataSet(DataSet& dataSet, int l)
{
  Eigen::MatrixXd X(dataSet.samples(), dataSet.inputs());
  Eigen::MatrixXd T(dataSet.samples(), dataSet.outputs());
  for(int n = 0; n < dataSet.samples(); n++)
  {
    tempInput = dataSet.getInstance(n).transpose();
    Eigen::MatrixXd* y = &tempInput;
    int i = 0;
    for(std::vector<Layer*>::iterator layer = layers.begin();
        layer != layers.end() && i < l; ++layer)
      (**layer).forwardPropagate(y, y, dropout);
    tempOutput = *y;
    X.row(n) = tempOutput;
    T.row(n) = dataSet.getTarget(n).transpose();
  }
  DirectStorageDataSet* transformedDataSet = new DirectStorageDataSet(&X, &T);
  return transformedDataSet;
}

void Net::initializeNetwork()
{
  P = parameters.size();
  tempInput.resize(1, infos[0].outputs());
  tempOutput.resize(1, infos.back().outputs());
  tempError.resize(1, infos.back().outputs());
  tempGradient.resize(P);
  parameterVector.resize(P);
  for(int p = 0; p < P; p++)
    parameterVector(p) = *parameters[p];
  initialized = true;
}

Net& Net::useDropout(bool activate)
{
  dropout = activate;
  return *this;
}

Net& Net::setRegularization(double l1Penalty, double l2Penalty,
                            double maxSquaredWeightNorm)
{
  regularization.l1Penalty = l1Penalty;
  regularization.l2Penalty = l2Penalty;
  regularization.maxSquaredWeightNorm = maxSquaredWeightNorm;
  return *this;
}

Net& Net::setErrorFunction(ErrorFunction errorFunction)
{
  this->errorFunction = errorFunction;
  return *this;
}

void Net::finishedIteration()
{
  bool dropout = this->dropout;
  this->dropout = false;
  if(trainSet)
    trainSet->finishIteration(*this);
  if(validSet)
    validSet->finishIteration(*this);
  this->dropout = dropout;
}

Eigen::VectorXd Net::operator()(const Eigen::VectorXd& x)
{
  tempInput = x.transpose();
  forwardPropagate();
  return tempOutput.transpose();
}

Eigen::MatrixXd Net::operator()(const Eigen::MatrixXd& x)
{
  tempInput = x;
  forwardPropagate();
  if(errorFunction == CE)
    OpenANN::softmax(tempOutput);
  return tempOutput;
}

unsigned int Net::dimension()
{
  return P;
}

unsigned int Net::examples()
{
  return N;
}

const Eigen::VectorXd& Net::currentParameters()
{
  return parameterVector;
}

void Net::setParameters(const Eigen::VectorXd& parameters)
{
  parameterVector = parameters;
  for(int p = 0; p < P; p++)
    *(this->parameters[p]) = parameters(p);
  for(std::vector<Layer*>::iterator layer = layers.begin();
      layer != layers.end(); ++layer)
    (**layer).updatedParameters();
}

bool Net::providesInitialization()
{
  return true;
}

void Net::initialize()
{
  OPENANN_CHECK(initialized);
  for(std::vector<Layer*>::iterator layer = layers.begin();
      layer != layers.end(); ++layer)
    (**layer).initializeParameters();
  for(int p = 0; p < P; p++)
    parameterVector(p) = *parameters[p];
}

double Net::error(unsigned int n)
{
  if(errorFunction == CE)
  {
    tempInput = trainSet->getInstance(n).transpose();
    forwardPropagate();
    return -(trainSet->getTarget(n).array() *
             ((tempOutput.transpose().array() + 1e-10).log())).sum();
  }
  else
  {
    tempInput = trainSet->getInstance(n).transpose();
    forwardPropagate();
    return (tempOutput.transpose() - trainSet->getTarget(n)).squaredNorm() / 2.0;
  }
}

double Net::error()
{
  double e = 0.0;
  for(int n = 0; n < N; n++)
    e += error(n) / (double) N;
  return e;
}

bool Net::providesGradient()
{
  return true;
}

Eigen::VectorXd Net::gradient(unsigned int n)
{
  generalErrorGradient(false, tempGradient, n);
  return tempGradient;
}

Eigen::VectorXd Net::gradient()
{
  generalErrorGradient(false, tempGradient);
  return tempGradient / (double) N;
}

void Net::errorGradient(int n, double& value, Eigen::VectorXd& grad)
{
  value = generalErrorGradient(true, grad, n);
}

void Net::errorGradient(double& value, Eigen::VectorXd& grad)
{
  value = generalErrorGradient(true, grad, -1) / N;
  grad /= N;
}

void Net::errorGradient(std::vector<int>::const_iterator startN,
                        std::vector<int>::const_iterator endN,
                        double& value, Eigen::VectorXd& grad)
{
  const int N = endN - startN;
  tempInput.conservativeResize(N, trainSet->inputs());
  Eigen::MatrixXd T(N, trainSet->outputs());
  int n = 0;
  for(std::vector<int>::const_iterator it = startN; it != endN; ++it, ++n)
  {
    tempInput.row(n) = trainSet->getInstance(*it);
    T.row(n) = trainSet->getTarget(*it);
  }
  forwardPropagate();
  tempError = tempOutput - T;
  if(errorFunction == CE)
    value = -(T.array() * ((tempOutput.array() + 1e-10).log())).sum();
  else
  {
    value = 0.0;
    for(int i = 0; i < tempError.rows(); i++)
      value += tempError.row(i).squaredNorm();
    value /= 2.0;
  }
  value /= (double) N;

  backpropagate();

  for(int p = 0; p < P; p++)
    grad(p) = *derivatives[p];
  grad /= N;
}

void Net::forwardPropagate()
{
  Eigen::MatrixXd* y = &tempInput;
  for(std::vector<Layer*>::iterator layer = layers.begin();
      layer != layers.end(); ++layer)
    (**layer).forwardPropagate(y, y, dropout);
  tempOutput = *y;
  OPENANN_CHECK_EQUALS(y->cols(), infos.back().outputs());
  if(errorFunction == CE)
    OpenANN::softmax(tempOutput);
}

void Net::backpropagate()
{
  Eigen::MatrixXd* e = &tempError;
  int l = L;
  for(std::vector<Layer*>::reverse_iterator layer = layers.rbegin();
      layer != layers.rend(); ++layer, --l)
    (**layer).backpropagate(e, e, l != 2);
}

double Net::generalErrorGradient(bool computeError, Eigen::VectorXd& g, int n)
{
  OPENANN_CHECK_EQUALS(g.rows(), dimension());

  const bool singleGradient = n >= 0;
  if(!singleGradient)
    g.setZero();

  const int start = singleGradient * n;
  const int end = singleGradient ? n + 1 : examples();

  double error = 0.0;
  for(int i = start; i < end; i++)
  {
    tempInput = trainSet->getInstance(i).transpose();
    forwardPropagate();
    tempError = tempOutput - trainSet->getTarget(i).transpose();

    if(computeError)
    {
      if(errorFunction == CE)
        error += -(trainSet->getTarget(i).array() *
                   ((tempOutput.transpose().array() + 1e-10).log())).sum();
      else
        error += tempError.squaredNorm() / 2.0;
    }

    backpropagate();

    if(singleGradient)
    {
      for(int p = 0; p < P; p++)
        g(p) = *derivatives[p];
    }
    else
    {
      for(int p = 0; p < P; p++)
        g(p) += *derivatives[p];
    }
  }

  return error;
}

}
