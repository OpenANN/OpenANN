#include <DeepNetwork.h>
#include <layers/Input.h>
#include <layers/AlphaBetaFilter.h>
#include <layers/FullyConnected.h>
#include <layers/Compressed.h>
#include <layers/Extreme.h>
#include <layers/Convolutional.h>
#include <layers/Subsampling.h>
#include <layers/MaxPooling.h>
#include <layers/LocalResponseNormalization.h>
#include <io/DirectStorageDataSet.h>
#include <optimization/IPOPCMAES.h>
#include <optimization/LMA.h>
#include <optimization/MBSGD.h>

namespace OpenANN {

DeepNetwork::DeepNetwork()
  : dataSet(0), testDataSet(0), deleteDataSet(false), deleteTestSet(false),
    errorFunction(SSE), dropout(false), initialized(false), N(0), L(0)
{
  layers.reserve(3);
  infos.reserve(3);
}

DeepNetwork::~DeepNetwork()
{
  if(deleteDataSet)
    delete dataSet;
  if(deleteTestSet)
    delete testDataSet;
  for(int i = 0; i < layers.size(); i++)
    delete layers[i];
  layers.clear();
}

DeepNetwork& DeepNetwork::inputLayer(int dim1, int dim2, int dim3, bool bias,
                                     fpt dropoutProbability)
{
  return addLayer(new Input(dim1, dim2, dim3, bias, dropoutProbability));
}

DeepNetwork& DeepNetwork::alphaBetaFilterLayer(fpt deltaT, fpt stdDev, bool bias)
{
  return addLayer(new AlphaBetaFilter(infos.back(), deltaT, bias, stdDev));
}

DeepNetwork& DeepNetwork::fullyConnectedLayer(int units, ActivationFunction act,
                                              fpt stdDev, bool bias,
                                              fpt dropoutProbability,
                                              fpt maxSquaredWeightNorm)
{
  return addLayer(new FullyConnected(infos.back(), units, bias, act, stdDev,
                                    dropoutProbability, maxSquaredWeightNorm));
}

DeepNetwork& DeepNetwork::compressedLayer(int units, int params,
                                          ActivationFunction act,
                                          const std::string& compression,
                                          fpt stdDev, bool bias,
                                          fpt dropoutProbability)
{
  return addLayer(new Compressed(infos.back(), units, params, bias, act,
                                compression, stdDev, dropoutProbability));
}

DeepNetwork& DeepNetwork::extremeLayer(int units, ActivationFunction act,
                                       fpt stdDev, bool bias)
{
  return addLayer(new Extreme(infos.back(), units, bias, act, stdDev));
}

DeepNetwork& DeepNetwork::convolutionalLayer(int featureMaps, int kernelRows,
                                             int kernelCols,
                                             ActivationFunction act,
                                             fpt stdDev, bool bias)
{
  return addLayer(new Convolutional(infos.back(), featureMaps, kernelRows,
                                   kernelCols, bias, act, stdDev));
}

DeepNetwork& DeepNetwork::subsamplingLayer(int kernelRows, int kernelCols,
                                           ActivationFunction act, fpt stdDev,
                                           bool bias)
{
  return addLayer(new Subsampling(infos.back(), kernelRows, kernelCols, bias, act, stdDev));
}

DeepNetwork& DeepNetwork::maxPoolingLayer(int kernelRows, int kernelCols, bool bias)
{
  return addLayer(new MaxPooling(infos.back(), kernelRows, kernelCols, bias));
}

DeepNetwork& DeepNetwork::localReponseNormalizationLayer(fpt k, int n,
                                                         fpt alpha, fpt beta,
                                                         bool bias)
{
  return addLayer(new LocalResponseNormalization(infos.back(), bias, k, n,
                                                 alpha, beta));
}

DeepNetwork& DeepNetwork::addLayer(Layer* layer)
{
    OPENANN_CHECK(layer != 0);

    OutputInfo info = layer->initialize(parameters, derivatives);
    layers.push_back(layer);
    infos.push_back(info);
    L++;
    return *this;
}

DeepNetwork& DeepNetwork::outputLayer(int units, ActivationFunction act,
                                      fpt stdDev)
{
  return fullyConnectedLayer(units, act, stdDev, false);
}

DeepNetwork& DeepNetwork::compressedOutputLayer(int units, int params,
                                                ActivationFunction act,
                                                const std::string& compression,
                                                fpt stdDev)
{
  return compressedLayer(units, params, act, compression, stdDev, false);
}

unsigned int DeepNetwork::numberOflayers()
{
  return L;
}

Layer& DeepNetwork::getLayer(unsigned int l)
{
  OPENANN_CHECK(l >= 0 && l < L);
  return *layers[l];
}

OutputInfo DeepNetwork::getOutputInfo(unsigned int l)
{
  OPENANN_CHECK(l >= 0 && l < L);
  return infos[l];
}

void DeepNetwork::initializeNetwork()
{
  P = parameters.size();
  tempInput.resize(infos[0].outputs() - infos[0].bias);
  tempOutput.resize(infos.back().outputs());
  tempError.resize(infos.back().outputs());
  tempGradient.resize(P);
  parameterVector.resize(P);
  for(int p = 0; p < P; p++)
    parameterVector(p) = *parameters[p];
  initialized = true;
}

Learner& DeepNetwork::trainingSet(Mt& trainingInput, Mt& trainingOutput)
{
  dataSet = new DirectStorageDataSet(trainingInput, trainingOutput);
  deleteDataSet = true;
  N = dataSet->samples();
  return *this;
}

Learner& DeepNetwork::trainingSet(DataSet& trainingSet)
{
  if(deleteDataSet)
    delete dataSet;
  dataSet = &trainingSet;
  deleteDataSet = false;
  N = dataSet->samples();
  return *this;
}

DeepNetwork& DeepNetwork::testSet(Mt& testInput, Mt& testOutput)
{
  if(deleteTestSet)
    delete testDataSet;
  testDataSet = new DirectStorageDataSet(testInput, testOutput);
  deleteTestSet = true;
  return *this;
}

DeepNetwork& DeepNetwork::testSet(DataSet& testSet)
{
  testDataSet = &testSet;
  deleteTestSet = false;
  return *this;
}

DeepNetwork& DeepNetwork::setErrorFunction(ErrorFunction errorFunction)
{
  this->errorFunction = errorFunction;
}

Vt DeepNetwork::train(Training algorithm, ErrorFunction errorFunction,
                      StoppingCriteria stop, bool reinitialize, bool dropout)
{
  if(reinitialize)
    initialize();
  Optimizer* opt;
  switch(algorithm)
  {
    case MINIBATCH_SGD:
      opt = new MBSGD;
      break;
#ifdef USE_GPL_LICENSE
    case BATCH_LMA:
      opt = new LMA;
      break;
#endif
    case BATCH_CMAES:
    default:
      opt = new IPOPCMAES;
      break;
  }
  this->dropout = dropout;
  opt->setOptimizable(*this);
  opt->setStopCriteria(stop);
  opt->optimize();
  Vt result = opt->result();
  delete opt;
  return result;
}

void DeepNetwork::finishedIteration()
{
  bool dropout = this->dropout;
  this->dropout = false;
  if(dataSet)
    dataSet->finishIteration(*this);
  if(testDataSet)
    testDataSet->finishIteration(*this);
  this->dropout = dropout;
}

Vt DeepNetwork::operator()(const Vt& x)
{
  tempInput = x;
  Vt* y = &tempInput;
  for(std::vector<Layer*>::iterator layer = layers.begin();
      layer != layers.end(); layer++)
    (**layer).forwardPropagate(y, y, dropout);
  tempOutput = *y;
  if(errorFunction == CE)
    OpenANN::softmax(tempOutput);
  return tempOutput;
}

unsigned int DeepNetwork::dimension()
{
  return P;
}

unsigned int DeepNetwork::examples()
{
  return N;
}

Vt DeepNetwork::currentParameters()
{
  return parameterVector;
}

void DeepNetwork::setParameters(const Vt& parameters)
{
  parameterVector = parameters;
  for(int p = 0; p < P; p++)
    *(this->parameters[p]) = parameters(p);
  for(std::vector<Layer*>::iterator layer = layers.begin();
      layer != layers.end(); layer++)
    (**layer).updatedParameters();
}

bool DeepNetwork::providesInitialization()
{
  return true;
}

void DeepNetwork::initialize()
{
    if(!initialized)
        initializeNetwork();

  for(std::vector<Layer*>::iterator layer = layers.begin();
      layer != layers.end(); layer++)
    (**layer).initializeParameters();
  for(int p = 0; p < P; p++)
    parameterVector(p) = *parameters[p];
}

fpt DeepNetwork::error(unsigned int i)
{
  fpt e = 0.0;
  if(errorFunction == CE)
  {
    tempOutput = (*this)(dataSet->getInstance(i));
    for(int f = 0; f < tempOutput.rows(); f++)
    {
      fpt out = tempOutput(f);
      if(out < 1e-45)
	out = 1e-45;
      e -= dataSet->getTarget(i)(f) * std::log(out);
    }
  }
  else
  {
    tempOutput = (*this)(dataSet->getInstance(i));
    tempError = tempOutput - dataSet->getTarget(i);
    e += tempError.dot(tempError);
  }
  return e / 2.0;
}

fpt DeepNetwork::error()
{
  fpt e = 0.0;
  for(int n = 0; n < N; n++)
    e += error(n);
  switch(errorFunction)
  {
    case SSE:
      return e;
    case MSE:
      return e / (fpt) N;
    default:
      return e;
  }
}

bool DeepNetwork::providesGradient()
{
  return true;
}

Vt DeepNetwork::gradient(unsigned int i)
{
  tempOutput = (*this)(dataSet->getInstance(i));
  tempError = tempOutput - dataSet->getTarget(i);
  Vt* e = &tempError;
  for(std::vector<Layer*>::reverse_iterator layer = layers.rbegin();
      layer != layers.rend(); layer++)
    (**layer).backpropagate(e, e);
  for(int i = 0; i < P; i++)
    tempGradient(i) = *derivatives[i];
  return tempGradient;
}

Vt DeepNetwork::gradient()
{
  tempGradient.fill(0.0);
  for(int n = 0; n < N; n++)
  {
    tempOutput = (*this)(dataSet->getInstance(n));
    tempError = tempOutput - dataSet->getTarget(n);
    Vt* e = &tempError;
    for(std::vector<Layer*>::reverse_iterator layer = layers.rbegin();
        layer != layers.rend(); layer++)
      (**layer).backpropagate(e, e);
    for(int i = 0; i < P; i++)
      tempGradient(i) += *derivatives[i];
  }
  switch(errorFunction)
  {
    case MSE:
      tempGradient /= (fpt) dimension();
    default:
      break;
  }
  return tempGradient;
}

void DeepNetwork::VJ(Vt& values, Mt& jacobian)
{
  for(unsigned n = 0; n < N; n++)
  {
    tempError = (*this)(dataSet->getInstance(n)) - dataSet->getTarget(n);
    values(n) = tempError.dot(tempError) / (fpt) 2.0;
    Vt* e = &tempError;
    for(std::vector<Layer*>::reverse_iterator layer = layers.rbegin();
        layer != layers.rend(); layer++)
      (**layer).backpropagate(e, e);
    for(int p = 0; p < P; p++)
      jacobian(n, p) = *derivatives[p];
  }
}

bool DeepNetwork::providesHessian()
{
  return false;
}

Mt DeepNetwork::hessian()
{
  return Mt::Identity(dimension(), dimension());
}

}
