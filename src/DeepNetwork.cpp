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
    errorFunction(errorFunction)
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

DeepNetwork& DeepNetwork::inputLayer(int dim1, int dim2, int dim3, bool bias)
{
  Layer* layer = new Input(dim1, dim2, dim3, bias);
  OutputInfo info = layer->initialize(parameters, derivatives);
  layers.push_back(layer);
  infos.push_back(info);
  return *this;
}

DeepNetwork& DeepNetwork::fullyConnectedLayer(int units, ActivationFunction act,
                                              fpt stdDev, bool bias)
{
  Layer* layer = new FullyConnected(infos.back(), units, bias, act, stdDev);
  OutputInfo info = layer->initialize(parameters, derivatives);
  layers.push_back(layer);
  infos.push_back(info);
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
  return *this;
}

DeepNetwork& DeepNetwork::outputLayer(int units, ActivationFunction act,
                                      fpt stdDev)
{
  return fullyConnectedLayer(units, act, stdDev, false);
}

Learner& DeepNetwork::trainingSet(Mt& trainingInput, Mt& trainingOutput)
{
  dataSet = new DirectStorageDataSet(trainingInput, trainingOutput);
  deleteDataSet = true;
  return *this;
}

Learner& DeepNetwork::trainingSet(DataSet& trainingSet)
{
  dataSet = &trainingSet;
  deleteDataSet = false;
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

unsigned int DeepNetwork::examples()
{
  if(dataSet)
    return dataSet->samples();
  else
    return 0;
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
  for(int n = 0; n < dataSet->samples(); n++)
    e += error(n);
  switch(errorFunction)
  {
    case SSE:
      return e;
    case MSE:
      return e / (fpt) dataSet->samples();
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
  Vt y = (*this)(dataSet->getInstance(i));
  Vt diff = y - dataSet->getTarget(i);
  Vt* e = &diff;
  for(std::vector<Layer*>::reverse_iterator layer = layers.rbegin();
      layer != layers.rend(); layer++)
  {
    (**layer).backpropagate(e, e);
  }
  Vt d(derivatives.size());
  std::list<fpt*>::iterator it = derivatives.begin();
  for(int i = 0; i < dimension(); i++, it++)
    d(i) = **it;
  return d;
}

Vt DeepNetwork::gradient()
{
  Vt g(dimension());
  g.fill(0.0);
  for(int n = 0; n < dataSet->samples(); n++)
  {
    g += gradient(n);
  }
  switch(errorFunction)
  {
    case MSE:
      g /= (fpt) dimension();
    default:
      break;
  }
  return g;
}

void DeepNetwork::VJ(Vt& values, Mt& jacobian)
{
  for(unsigned n = 0; n < dataSet->samples(); n++)
  {
    Vt diff = (*this)(dataSet->getInstance(n)) - dataSet->getTarget(n);
    Eigen::Matrix<fpt, 1, 1> err = diff.transpose() * diff / (fpt) 2.0;
    values(n) = err(0, 0);
    Vt* e = &diff;
    for(std::vector<Layer*>::reverse_iterator layer = layers.rbegin();
        layer != layers.rend(); layer++)
    {
      (**layer).backpropagate(e, e);
    }
    Vt d(derivatives.size());
    std::list<fpt*>::iterator it = derivatives.begin();
    for(int i = 0; i < dimension(); i++, it++)
      d(i) = **it;
    jacobian.row(n) = d;
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
