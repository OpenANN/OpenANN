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
#include <OpenANN/io/DirectStorageDataSet.h>
#include <OpenANN/optimization/IPOPCMAES.h>
#include <OpenANN/optimization/LMA.h>
#include <OpenANN/optimization/MBSGD.h>

namespace OpenANN {

Net::Net()
  : dataSet(0), testDataSet(0), deleteDataSet(false), deleteTestSet(false),
    errorFunction(SSE), dropout(false), initialized(false), N(0), L(0)
{
  layers.reserve(3);
  infos.reserve(3);
}

Net::~Net()
{
  if(deleteDataSet)
    delete dataSet;
  if(deleteTestSet)
    delete testDataSet;
  for(int i = 0; i < layers.size(); i++)
    delete layers[i];
  layers.clear();
}

Net& Net::inputLayer(int dim1, int dim2, int dim3, bool bias,
                     double dropoutProbability)
{
  return addLayer(new Input(dim1, dim2, dim3, bias, dropoutProbability));
}

Net& Net::alphaBetaFilterLayer(double deltaT, double stdDev, bool bias)
{
  return addLayer(new AlphaBetaFilter(infos.back(), deltaT, bias, stdDev));
}

Net& Net::fullyConnectedLayer(int units, ActivationFunction act, double stdDev,
                              bool bias, double dropoutProbability,
                              double maxSquaredWeightNorm)
{
  return addLayer(new FullyConnected(infos.back(), units, bias, act, stdDev,
                                    dropoutProbability, maxSquaredWeightNorm));
}

Net& Net::compressedLayer(int units, int params, ActivationFunction act,
                          const std::string& compression, double stdDev,
                          bool bias, double dropoutProbability)
{
  return addLayer(new Compressed(infos.back(), units, params, bias, act,
                                compression, stdDev, dropoutProbability));
}

Net& Net::extremeLayer(int units, ActivationFunction act, double stdDev,
                       bool bias)
{
  return addLayer(new Extreme(infos.back(), units, bias, act, stdDev));
}

Net& Net::convolutionalLayer(int featureMaps, int kernelRows, int kernelCols,
                             ActivationFunction act, double stdDev, bool bias)
{
  return addLayer(new Convolutional(infos.back(), featureMaps, kernelRows,
                                   kernelCols, bias, act, stdDev));
}

Net& Net::subsamplingLayer(int kernelRows, int kernelCols,
                           ActivationFunction act, double stdDev, bool bias)
{
  return addLayer(new Subsampling(infos.back(), kernelRows, kernelCols, bias, act, stdDev));
}

Net& Net::maxPoolingLayer(int kernelRows, int kernelCols, bool bias)
{
  return addLayer(new MaxPooling(infos.back(), kernelRows, kernelCols, bias));
}

Net& Net::localReponseNormalizationLayer(double k, int n, double alpha, double beta,
                                         bool bias)
{
  return addLayer(new LocalResponseNormalization(infos.back(), bias, k, n,
                                                 alpha, beta));
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

Net& Net::outputLayer(int units, ActivationFunction act, double stdDev)
{
  return fullyConnectedLayer(units, act, stdDev, false);
}

Net& Net::compressedOutputLayer(int units, int params, ActivationFunction act,
                                const std::string& compression, double stdDev)
{
  return compressedLayer(units, params, act, compression, stdDev, false);
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

void Net::initializeNetwork()
{
  P = parameters.size();
  tempInput.resize(infos[0].outputs() - infos[0].bias);
  tempOutput.resize(infos.back().outputs());
  tempError.resize(infos.back().outputs());
  tempGradient.resize(P);
  parameterVector.resize(P);
  initialized = true;
}

Net& Net::useDropout(bool activate)
{
  dropout = activate;
}

Learner& Net::trainingSet(Eigen::MatrixXd& trainingInput, Eigen::MatrixXd& trainingOutput)
{
  dataSet = new DirectStorageDataSet(trainingInput, trainingOutput);
  deleteDataSet = true;
  N = dataSet->samples();
  return *this;
}

Learner& Net::trainingSet(DataSet& trainingSet)
{
  if(deleteDataSet)
    delete dataSet;
  dataSet = &trainingSet;
  deleteDataSet = false;
  N = dataSet->samples();
  return *this;
}

Net& Net::testSet(Eigen::MatrixXd& testInput, Eigen::MatrixXd& testOutput)
{
  if(deleteTestSet)
    delete testDataSet;
  testDataSet = new DirectStorageDataSet(testInput, testOutput);
  deleteTestSet = true;
  return *this;
}

Net& Net::testSet(DataSet& testSet)
{
  testDataSet = &testSet;
  deleteTestSet = false;
  return *this;
}

Net& Net::setErrorFunction(ErrorFunction errorFunction)
{
  this->errorFunction = errorFunction;
}

void Net::finishedIteration()
{
  bool dropout = this->dropout;
  this->dropout = false;
  if(dataSet)
    dataSet->finishIteration(*this);
  if(testDataSet)
    testDataSet->finishIteration(*this);
  this->dropout = dropout;
}

Eigen::VectorXd Net::operator()(const Eigen::VectorXd& x)
{
  tempInput = x;
  Eigen::VectorXd* y = &tempInput;
  for(std::vector<Layer*>::iterator layer = layers.begin();
      layer != layers.end(); layer++)
    (**layer).forwardPropagate(y, y, dropout);
  tempOutput = *y;
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

Eigen::VectorXd Net::currentParameters()
{
  return parameterVector;
}

void Net::setParameters(const Eigen::VectorXd& parameters)
{
  parameterVector = parameters;
  for(int p = 0; p < P; p++)
    *(this->parameters[p]) = parameters(p);
  for(std::vector<Layer*>::iterator layer = layers.begin();
      layer != layers.end(); layer++)
    (**layer).updatedParameters();
}

bool Net::providesInitialization()
{
  return true;
}

void Net::initialize()
{
    if(!initialized)
        initializeNetwork();

  for(std::vector<Layer*>::iterator layer = layers.begin();
      layer != layers.end(); layer++)
    (**layer).initializeParameters();
  for(int p = 0; p < P; p++)
    parameterVector(p) = *parameters[p];
}

double Net::error(unsigned int i)
{
  double e = 0.0;
  if(errorFunction == CE)
  {
    tempOutput = (*this)(dataSet->getInstance(i));
    for(int f = 0; f < tempOutput.rows(); f++)
    {
      double out = tempOutput(f);
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

double Net::error()
{
  double e = 0.0;
  for(int n = 0; n < N; n++)
    e += error(n);
  switch(errorFunction)
  {
    case SSE:
      return e;
    case MSE:
      return e / (double) N;
    default:
      return e;
  }
}

double Net::errorFromDataSet(DataSet& dataset)
{
    double e = 0.0;
    for(int n = 0; n < dataset.samples(); ++n) {
        double e_n  = 0.0;

        tempOutput = (*this)(dataset.getInstance(n));

        if(errorFunction == CE) 
        {
            for(int f = 0; f < tempOutput.rows(); ++f)
                e_n -= dataset.getTarget(n)(f) * std::log(tempOutput(f));
        }
        else 
        {
            tempError = tempOutput - dataset.getTarget(n);
            e_n += tempError.dot(tempError);
        }

        e += (e_n / 2);
    }

    switch(errorFunction) 
    {
        case SSE:
            return e;
        case MSE:
            return e / (double) N;
        default:
            return e;
    }
}

bool Net::providesGradient()
{
  return true;
}

Eigen::VectorXd Net::gradient(unsigned int i)
{
  tempOutput = (*this)(dataSet->getInstance(i));
  tempError = tempOutput - dataSet->getTarget(i);
  Eigen::VectorXd* e = &tempError;
  for(std::vector<Layer*>::reverse_iterator layer = layers.rbegin();
      layer != layers.rend(); layer++)
    (**layer).backpropagate(e, e);
  for(int i = 0; i < P; i++)
    tempGradient(i) = *derivatives[i];
  return tempGradient;
}

Eigen::VectorXd Net::gradient()
{
  tempGradient.fill(0.0);
  for(int n = 0; n < N; n++)
  {
    tempOutput = (*this)(dataSet->getInstance(n));
    tempError = tempOutput - dataSet->getTarget(n);
    Eigen::VectorXd* e = &tempError;
    for(std::vector<Layer*>::reverse_iterator layer = layers.rbegin();
        layer != layers.rend(); layer++)
      (**layer).backpropagate(e, e);
    for(int i = 0; i < P; i++)
      tempGradient(i) += *derivatives[i];
  }
  switch(errorFunction)
  {
    case MSE:
      tempGradient /= (double) dimension();
    default:
      break;
  }
  return tempGradient;
}

void Net::VJ(Eigen::VectorXd& values, Eigen::MatrixXd& jacobian)
{
  for(unsigned n = 0; n < N; n++)
  {
    tempError = (*this)(dataSet->getInstance(n)) - dataSet->getTarget(n);
    values(n) = tempError.dot(tempError) / 2.0;
    Eigen::VectorXd* e = &tempError;
    for(std::vector<Layer*>::reverse_iterator layer = layers.rbegin();
        layer != layers.rend(); layer++)
      (**layer).backpropagate(e, e);
    for(int p = 0; p < P; p++)
      jacobian(n, p) = *derivatives[p];
  }
}

bool Net::providesHessian()
{
  return false;
}

Eigen::MatrixXd Net::hessian()
{
  return Eigen::MatrixXd::Identity(dimension(), dimension());
}

}
