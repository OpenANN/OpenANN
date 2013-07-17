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
#include <OpenANN/ErrorFunctions.h>
#include <OpenANN/io/DirectStorageDataSet.h>
#include <OpenANN/optimization/IPOPCMAES.h>
#include <OpenANN/optimization/LMA.h>
#include <OpenANN/optimization/MBSGD.h>
#include <OpenANN/util/OpenANNException.h>
#include <fstream>

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
  architecture << "input " << dim1 << " " << dim2 << " " << dim3 << " ";
  return addLayer(new Input(dim1, dim2, dim3));
}

Net& Net::alphaBetaFilterLayer(double deltaT, double stdDev)
{
  architecture << "alpha_beta_filter " << deltaT << " " << stdDev << " ";
  return addLayer(new AlphaBetaFilter(infos.back(), deltaT, stdDev));
}

Net& Net::fullyConnectedLayer(int units, ActivationFunction act, double stdDev,
                              bool bias)
{
  architecture << "fully_connected " << units << " " << (int) act << " "
      << stdDev << " " << bias << " ";
  return addLayer(new FullyConnected(infos.back(), units, bias, act, stdDev,
                                     regularization));
}

Net& Net::restrictedBoltzmannMachineLayer(int H, int cdN, double stdDev,
                                          bool backprop)
{
  architecture << "rbm " << H << " " << cdN << " " << stdDev << " "
      << backprop << " ";
  return addLayer(new RBM(infos.back().outputs(), H, cdN, stdDev,
                          backprop, regularization));
}

Net& Net::compressedLayer(int units, int params, ActivationFunction act,
                          const std::string& compression, double stdDev,
                          bool bias)
{
  architecture << "compressed " << units << " " << params << " " << (int) act
      << " " << compression << " " << stdDev << " " << bias << " ";
  return addLayer(new Compressed(infos.back(), units, params, bias, act,
                                 compression, stdDev, regularization));
}

Net& Net::extremeLayer(int units, ActivationFunction act, double stdDev,
                       bool bias)
{
  architecture << "extreme " << units << " " << (int) act << " " << stdDev
      << " " << bias << " ";
  return addLayer(new Extreme(infos.back(), units, bias, act, stdDev));
}

Net& Net::intrinsicPlasticityLayer(double targetMean, double stdDev)
{
  architecture << "intrinsic_plasticity " << targetMean << " " << stdDev << " ";
  return addLayer(new IntrinsicPlasticity(infos.back().outputs(), targetMean,
                                          stdDev));
}

Net& Net::convolutionalLayer(int featureMaps, int kernelRows, int kernelCols,
                             ActivationFunction act, double stdDev, bool bias)
{
  architecture << "convolutional " << featureMaps << " " << kernelRows << " "
      << kernelCols << " " << (int) act << " " << stdDev << " " << bias << " ";
  return addLayer(new Convolutional(infos.back(), featureMaps, kernelRows,
                                    kernelCols, bias, act, stdDev, regularization));
}

Net& Net::subsamplingLayer(int kernelRows, int kernelCols,
                           ActivationFunction act, double stdDev, bool bias)
{
  architecture << "subsampling " << kernelRows << " " << kernelCols << " "
      << (int) act << " " << stdDev << " " << bias << " ";
  return addLayer(new Subsampling(infos.back(), kernelRows, kernelCols, bias,
                                  act, stdDev, regularization));
}

Net& Net::maxPoolingLayer(int kernelRows, int kernelCols)
{
  architecture << "max_pooling " << kernelRows << " " << kernelCols << " ";
  return addLayer(new MaxPooling(infos.back(), kernelRows, kernelCols));
}

Net& Net::localReponseNormalizationLayer(double k, int n, double alpha,
                                         double beta)
{
  architecture << "local_response_normalization " << k << " " << n << " "
      << alpha << " " << beta << " ";
  return addLayer(new LocalResponseNormalization(infos.back(), k, n, alpha,
                                                 beta));
}

Net& Net::dropoutLayer(double dropoutProbability)
{
  architecture << "dropout " << dropoutProbability << " ";
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
  architecture << "output " << units << " " << (int) act << " " << stdDev
      << " " << bias << " ";
  addLayer(new FullyConnected(infos.back(), units, bias, act, stdDev,
                              regularization));
  initializeNetwork();
  return *this;
}

Net& Net::compressedOutputLayer(int units, int params, ActivationFunction act,
                                const std::string& compression, double stdDev,
                                bool bias)
{
  architecture << "compressed_output " << units << " " << params << " "
      << (int) act << " " << compression << " " << stdDev << " " << bias << " ";
  addLayer(new Compressed(infos.back(), units, params, bias, act, compression,
                          stdDev, regularization));
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

void Net::save(const std::string& fileName)
{
  std::ofstream file(fileName.c_str());
  if(!file.is_open())
    throw OpenANNException("Could not open '" + fileName + "'.'");
  save(file);
  file.close();
}

void Net::save(std::ostream& stream)
{
  stream << architecture.str() << "parameters " << currentParameters();
}

void Net::load(const std::string& fileName)
{
  std::ifstream file(fileName.c_str());
  if(!file.is_open())
    throw OpenANNException("Could not open '" + fileName + "'.'");
  load(file);
  file.close();
}

void Net::load(std::istream& stream)
{
  std::string type;
  while(!stream.eof())
  {
    stream >> type;
    if(type == "input")
    {
      int dim1, dim2, dim3;
      stream >> dim1 >> dim2 >> dim3;
      OPENANN_DEBUG << "input " << dim1 << " " << dim2 << " " << dim3;
      inputLayer(dim1, dim2, dim3);
    }
    else if(type == "alpha_beta_filter")
    {
      double deltaT, stdDev;
      stream >> deltaT >> stdDev;
      OPENANN_DEBUG << "alpha_beta_filter" << deltaT << " " << stdDev;
      alphaBetaFilterLayer(deltaT, stdDev);
    }
    else if(type == "fully_connected")
    {
      int units;
      int act;
      double stdDev;
      bool bias;
      stream >> units >> act >> stdDev >> bias;
      OPENANN_DEBUG << "fully_connected " << units << " " << act << " "
          << stdDev << " " << bias;
      fullyConnectedLayer(units, (ActivationFunction) act, stdDev, bias);
    }
    else if(type == "rbm")
    {
      int H;
      int cdN;
      double stdDev;
      bool backprop;
      stream >> H >> cdN >> stdDev >> backprop;
      OPENANN_DEBUG << "rbm " << H << " " << cdN << " " << stdDev << " "
          << backprop;
      restrictedBoltzmannMachineLayer(H, cdN, stdDev, backprop);
    }
    else if(type == "compressed")
    {
      int units;
      int params;
      int act;
      std::string compression;
      double stdDev;
      bool bias;
      stream >> units >> params >> act >> compression >> stdDev >> bias;
      OPENANN_DEBUG << "compressed " << units << " " << params << " " << act
          << " " << compression << " " << stdDev << " " << bias;
      compressedLayer(units, params, (ActivationFunction) act, compression,
                      stdDev, bias);
    }
    else if(type == "extreme")
    {
      int units;
      int act;
      double stdDev;
      bool bias;
      stream >> units >> act >> stdDev >> bias;
      OPENANN_DEBUG << "extreme " << units << " " << act << " " << stdDev
          << " " << bias;
      extremeLayer(units, (ActivationFunction) act, stdDev, bias);
    }
    else if(type == "intrinsic_plasticity")
    {
      double targetMean;
      double stdDev;
      stream >> targetMean >> stdDev;
      OPENANN_DEBUG << "intrinsic_plasticity " << targetMean << " " << stdDev;
      intrinsicPlasticityLayer(targetMean, stdDev);
    }
    else if(type == "convolutional")
    {
      int featureMaps, kernelRows, kernelCols, act;
      double stdDev;
      bool bias;
      stream >> featureMaps >> kernelRows >> kernelCols >> act >> stdDev >> bias;
      OPENANN_DEBUG << "convolutional " << featureMaps << " " << kernelRows
          << " " << kernelCols << " " << act << " " << stdDev << " " << bias;
      convolutionalLayer(featureMaps, kernelRows, kernelCols,
                         (ActivationFunction) act, stdDev, bias);
    }
    else if(type == "subsampling")
    {
      int kernelRows, kernelCols, act;
      double stdDev;
      bool bias;
      stream >> kernelRows >> kernelCols >> act >> stdDev >> bias;
      OPENANN_DEBUG << "subsampling " << kernelRows << " " << kernelCols
          << " " << act << " " << stdDev << " " << bias;
      subsamplingLayer(kernelRows, kernelCols, (ActivationFunction) act,
                       stdDev, bias);
    }
    else if(type == "max_pooling")
    {
      int kernelRows, kernelCols;
      stream >> kernelRows >> kernelCols;
      OPENANN_DEBUG << "max_pooling " << kernelRows << " " << kernelCols;
      maxPoolingLayer(kernelRows, kernelCols);
    }
    else if(type == "local_response_normalization")
    {
      double k, alpha, beta;
      int n;
      stream >> k >> n >> alpha >> beta;
      OPENANN_DEBUG << "local_response_normalization " << k << " " << n << " "
          << alpha << " " << beta;
      localReponseNormalizationLayer(k, n, alpha, beta);
    }
    else if(type == "dropout")
    {
      double dropoutProbability;
      stream >> dropoutProbability;
      OPENANN_DEBUG << "dropout " << dropoutProbability;
      dropoutLayer(dropoutProbability);
    }
    else if(type == "output")
    {
      int units;
      int act;
      double stdDev;
      bool bias;
      stream >> units >> act >> stdDev >> bias;
      OPENANN_DEBUG << "output " << units << " " << act << " " << stdDev
          << " " << bias;
      outputLayer(units, (ActivationFunction) act, stdDev, bias);
    }
    else if(type == "compressed_output")
    {
      int units;
      int params;
      int act;
      std::string compression;
      double stdDev;
      bool bias;
      stream >> units >> params >> act >> compression >> stdDev >> bias;
      OPENANN_DEBUG << "compressed_output " << units << " " << params << " "
          << act << " " << compression << " " << stdDev << " " << bias;
      compressedOutputLayer(units, params, (ActivationFunction) act,
                            compression, stdDev, bias);
    }
    else if(type == "error_function")
    {
      int errorFunction;
      stream >> errorFunction;
      OPENANN_DEBUG << "error_function " << errorFunction;
      setErrorFunction((ErrorFunction) errorFunction);
    }
    else if(type == "regularization")
    {
      double l1Penalty, l2Penalty, maxSquaredWeightNorm;
      stream >> l1Penalty >> l2Penalty >> maxSquaredWeightNorm;
      OPENANN_DEBUG << "regularization " << l1Penalty << " " << l2Penalty
          << " " << maxSquaredWeightNorm;
      setRegularization(l1Penalty, l2Penalty, maxSquaredWeightNorm);
    }
    else if(type == "parameters")
    {
      double p = 0.0;
      for(int i = 0; i < dimension(); i++)
        stream >> parameterVector(i);
      setParameters(parameterVector);
    }
    else
    {
      throw OpenANNException("Unknown layer type: '" + type + "'.");
    }
  }
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
  architecture << "regularization " << l1Penalty << " " << l2Penalty << " "
      << maxSquaredWeightNorm << " ";
  regularization.l1Penalty = l1Penalty;
  regularization.l2Penalty = l2Penalty;
  regularization.maxSquaredWeightNorm = maxSquaredWeightNorm;
  return *this;
}

Net& Net::setErrorFunction(ErrorFunction errorFunction)
{
  architecture << "error_function " << (int) errorFunction << " ";
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
  tempInput = trainSet->getInstance(n).transpose();
  forwardPropagate();
  if(errorFunction == CE)
    return crossEntropy(tempOutput, trainSet->getTarget(n).transpose());
  else
    return meanSquaredError(tempOutput - trainSet->getTarget(n).transpose());
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
  std::vector<int> indices;
  indices.push_back(n);
  double error;
  errorGradient(indices.begin(), indices.end(), error, tempGradient);
  return tempGradient;
}

Eigen::VectorXd Net::gradient()
{
  std::vector<int> indices;
  indices.reserve(N);
  for(int n = 0; n < N; n++)
    indices.push_back(n);
  double error;
  errorGradient(indices.begin(), indices.end(), error, tempGradient);
  return tempGradient;
}

void Net::errorGradient(int n, double& value, Eigen::VectorXd& grad)
{
  std::vector<int> indices;
  indices.push_back(n);
  errorGradient(indices.begin(), indices.end(), value, grad);
}

void Net::errorGradient(double& value, Eigen::VectorXd& grad)
{
  std::vector<int> indices;
  for(int n = 0; n < N; n++)
    indices.push_back(n);
  errorGradient(indices.begin(), indices.end(), value, grad);
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
  value = errorFunction == CE ? crossEntropy(tempOutput, T) :
      meanSquaredError(tempError);
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

}
