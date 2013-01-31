#include <DeepNetwork.h>
#include <layers/Input.h>
#include <layers/FullyConnected.h>
#include <layers/Convolutional.h>
#include <layers/Subsampling.h>
#include <io/DirectStorageDataSet.h>
#include <optimization/IPOPCMAES.h>
#include <optimization/LMA.h>
#include <optimization/SGD.h>
#include <Random.h>

namespace OpenANN {

DeepNetwork::DeepNetwork(ErrorFunction errorFunction)
  : debugLogger(Logger::NONE), deleteDataSet(false),
    errorFunction(errorFunction), initialized(false), N(0), L(0)
{
  layers.reserve(3);
  infos.reserve(3);
}

DeepNetwork::~DeepNetwork()
{
  if(deleteDataSet)
    delete dataSet;
  for(int i = 0; i < layers.size(); i++)
    delete layers[i];
  layers.clear();
}

DeepNetwork& DeepNetwork::inputLayer(int dim1, int dim2, int dim3, bool bias)
{
  Layer* layer = new Input(dim1, dim2, dim3, bias);
  OutputInfo info = layer->initialize(parameters, derivatives);
  layers.push_back(layer);
  infos.push_back(info);
  L++;
  return *this;
}

DeepNetwork& DeepNetwork::fullyConnectedLayer(int units, ActivationFunction act,
                                              fpt stdDev, bool bias)
{
  Layer* layer = new FullyConnected(infos.back(), units, bias, act, stdDev);
  OutputInfo info = layer->initialize(parameters, derivatives);
  layers.push_back(layer);
  infos.push_back(info);
  L++;
  return *this;
}

DeepNetwork& DeepNetwork::convolutionalLayer(int featureMaps, int kernelRows,
                                             int kernelCols,
                                             ActivationFunction act,
                                             fpt stdDev, bool bias)
{
  Layer* layer = new Convolutional(infos.back(), featureMaps, kernelRows,
                                   kernelCols, bias, act, stdDev);
  OutputInfo info = layer->initialize(parameters, derivatives);
  layers.push_back(layer);
  infos.push_back(info);
  L++;
  return *this;
}

DeepNetwork& DeepNetwork::subsamplingLayer(int kernelRows, int kernelCols,
                                           ActivationFunction act, fpt stdDev,
                                           bool bias)
{
  Layer* layer = new Subsampling(infos.back(), kernelRows, kernelCols, bias,
                                 act, stdDev);
  OutputInfo info = layer->initialize(parameters, derivatives);
  layers.push_back(layer);
  infos.push_back(info);
  L++;
  return *this;
}

DeepNetwork& DeepNetwork::outputLayer(int units, ActivationFunction act,
                                      fpt stdDev)
{
  fullyConnectedLayer(units, act, stdDev, false);
  L++;

  tempInput.resize(infos[0].outputs()-infos[0].bias);
  tempOutput.resize(units);
  tempError.resize(units);
  tempParameters.resize(parameters.size());
  tempParametersSum.resize(parameters.size());
  P = parameters.size();
  initialized = true;

  return *this;
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
  dataSet = &trainingSet;
  deleteDataSet = false;
  N = dataSet->samples();
  return *this;
}

Vt DeepNetwork::train(Training algorithm, StopCriteria stop, bool reinitialize)
{
  if(reinitialize)
    initialize();
  Optimizer* opt;
  switch(algorithm)
  {
    case BATCH_SGD:
      opt = new SGD;
      break;
#ifdef USE_GPL_LICENSE
    case BATCH_LMA:
      opt = new LMA(true);
      break;
#endif
    case BATCH_CMAES:
    default:
      opt = new IPOPCMAES;
      break;
  }
  opt->setOptimizable(*this);
  opt->setStopCriteria(stop);
  opt->optimize();
  Vt result = opt->result();
  delete opt;
  return result;
}

void DeepNetwork::finishedIteration()
{
  if(dataSet)
    dataSet->finishIteration(*this);
}

Vt DeepNetwork::operator()(const Vt& x)
{
  tempInput = x;
  Vt* y = &tempInput;
  for(std::vector<Layer*>::iterator layer = layers.begin();
      layer != layers.end(); layer++)
    (**layer).forwardPropagate(y, y);
  return *y;
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
  std::vector<fpt*>::iterator it = parameters.begin();
  for(int p = 0; p < P; p++, it++)
    tempParameters(p) = **it;
  return tempParameters;
}

void DeepNetwork::setParameters(const Vt& parameters)
{
  std::vector<fpt*>::iterator it = this->parameters.begin();
  for(int p = 0; p < P; p++, it++)
    **it = parameters(p);
}

bool DeepNetwork::providesInitialization()
{
  return true;
}

void DeepNetwork::initialize()
{
  RandomNumberGenerator rng;
  for(std::vector<fpt*>::iterator it = parameters.begin();
      it != parameters.end(); it++)
    **it = rng.sampleNormalDistribution<fpt>() * 0.05; // TODO remove magic number
}

fpt DeepNetwork::error(unsigned int i)
{
  fpt e = 0.0;
  if(errorFunction == CE)
  {
    tempOutput = (*this)(dataSet->getInstance(i));
    OpenANN::softmax(tempOutput);
    for(int f = 0; f < tempOutput.rows(); f++)
      e -= dataSet->getTarget(i)(f) * std::log(tempOutput(f));
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
  std::vector<fpt*>::iterator it = derivatives.begin();
  for(int i = 0; i < P; i++, it++)
    tempParameters(i) = **it;
  return tempParameters;
}

Vt DeepNetwork::gradient()
{
  tempParametersSum.fill(0.0);
  for(int n = 0; n < N; n++)
    tempParametersSum += gradient(n);
  switch(errorFunction)
  {
    case MSE:
      tempParametersSum /= (fpt) dimension();
    default:
      break;
  }
  return tempParametersSum;
}

void DeepNetwork::VJ(Vt& values, Mt& jacobian)
{
  for(unsigned n = 0; n < N; n++)
  {
    tempError = (*this)(dataSet->getInstance(n)) - dataSet->getTarget(n);
    values(n) = tempError.dot(tempError) / (fpt) 2.0;
    Vt* e = &tempOutput;
    for(std::vector<Layer*>::reverse_iterator layer = layers.rbegin();
        layer != layers.rend(); layer++)
      (**layer).backpropagate(e, e);
    std::vector<fpt*>::iterator it = derivatives.begin();
    for(int i = 0; i < P; i++, it++)
      tempParameters(i) = **it;
    jacobian.row(n) = tempParameters;
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
