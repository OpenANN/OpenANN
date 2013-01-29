#include <DeepNetwork.h>
#include <layers/FullyConnected.h>
#include <layers/Convolutional.h>
#include <layers/Subsampling.h>
#include <io/DirectStorageDataSet.h>
#include <Random.h>

namespace OpenANN {

DeepNetwork::DeepNetwork(ErrorFunction errorFunction)
  : deleteDataSet(false), errorFunction(errorFunction)
{
}

DeepNetwork::~DeepNetwork()
{
  if(deleteDataSet)
    delete dataSet;
  for(int i = 0; i < layers.size(); i++)
    delete layers[i];
  layers.clear();
}

DeepNetwork& DeepNetwork::inputLayer()
{
}

DeepNetwork& DeepNetwork::fullyConnectedLayer(int units, bool bias,
                                              ActivationFunction act,
                                              fpt stdDev)
{
  Layer* layer = new FullyConnected(infos.back(), units, bias, act, stdDev);
  OutputInfo info = layer->initialize(parameters, derivatives);
  layers.push_back(layer);
  infos.push_back(info);
}

DeepNetwork& DeepNetwork::convolutionalLayer(int featureMaps, int kernelRows,
                                             int kernelCols, bool bias,
                                             ActivationFunction act,
                                             fpt stdDev)
{
  Layer* layer = new Convolutional(infos.back(), featureMaps, kernelRows,
                                   kernelCols, bias, act, stdDev);
  OutputInfo info = layer->initialize(parameters, derivatives);
  layers.push_back(layer);
  infos.push_back(info);
}

DeepNetwork& DeepNetwork::subsamplingLayer(int kernelRows, int kernelCols,
                                           bool bias, ActivationFunction act,
                                           fpt stdDev)
{
  Layer* layer = new Subsampling(infos.back(), kernelRows, kernelCols, bias,
                                 act, stdDev);
  OutputInfo info = layer->initialize(parameters, derivatives);
  layers.push_back(layer);
  infos.push_back(info);
}

DeepNetwork& DeepNetwork::outputLayer()
{
}

Learner& DeepNetwork::trainingSet(Mt& trainingInput, Mt& trainingOutput)
{
  dataSet = new DirectStorageDataSet(trainingInput, trainingOutput);
  deleteDataSet = true;
}

Learner& DeepNetwork::trainingSet(DataSet& trainingSet)
{
  dataSet = &trainingSet;
  deleteDataSet = false;
}

Vt DeepNetwork::operator()(const Vt& x)
{
  Vt in = x;
  Vt* y = &in;
  for(std::vector<Layer*>::iterator layer = layers.begin();
      layer != layers.end(); layer++)
  {
    (**layer).forwardPropagate(y, y);
  }
  return *y;
}

unsigned int DeepNetwork::dimension()
{
  return parameters.size();
}

Vt DeepNetwork::currentParameters()
{
  Vt p(parameters.size());
  std::list<fpt*>::iterator it = parameters.begin();
  for(int i = 0; i < dimension(); i++, it++)
    p(i) = **it;
  return p;
}

void DeepNetwork::setParameters(const Vt& parameters)
{
  std::list<fpt*>::iterator it = this->parameters.begin();
  for(int i = 0; i < dimension(); i++, it++)
    **it = parameters(i);
}

bool DeepNetwork::providesInitialization()
{
  return true;
}

void DeepNetwork::initialize()
{
  RandomNumberGenerator rng;
  std::list<fpt*>::iterator it = parameters.begin();
  for(int i = 0; i < dimension(); i++, it++)
    **it = rng.sampleNormalDistribution<fpt>() * 0.05; // TODO remove magic number
}

fpt DeepNetwork::error(unsigned int i)
{
  fpt e = 0.0;
  if(errorFunction == CE)
  {
    Vt y = (*this)(dataSet->getInstance(i));
    OpenANN::softmax(y);
    for(int f = 0; f < y.rows(); f++)
      e -= dataSet->getTarget(i)(f) * std::log(y(f));
  }
  else
  {
    Vt y = (*this)(dataSet->getInstance(i));
    const Vt diff = y - dataSet->getTarget(i);
    e += diff.dot(diff);
  }
  return e / 2.0;
}

fpt DeepNetwork::error()
{
  fpt e = 0.0;
  for(int n = 0; n < dimension(); n++)
    e += error(n);
  switch(errorFunction)
  {
    case SSE:
      return e / 2.0;
    case MSE:
      return e / (fpt) dimension();
    default:
      return e;
  }
}

bool DeepNetwork::providesGradient()
{
  return true;
}

Vt DeepNetwork::gradient()
{
  // TODO implement
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
