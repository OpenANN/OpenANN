#include <MLP.h>
#include <optimization/IPOPCMAES.h>
#include <optimization/LMA.h>
#include <optimization/SGD.h>
#include <io/DirectStorageDataSet.h>
#include <AssertionMacros.h>
#include <ActivationFunctions.h>
#include <sstream>

namespace OpenANN {

MLP::MLP(Logger::Target errorTarget, Logger::Target parameterTarget)
  : layerState(NO_LAYER), initializationState(UNINITIALIZED),
    errorFunction(NO_E_DEFINED),
    trainingData(0), testData(0), deleteDataSetOnDestruction(false),
    optimizer(0), reinitialize(true), iteration(0),
    N(-1),
    errorLogger(errorTarget, "mlp-error"),
    parameterLogger(parameterTarget, "mlp-parameters")
{
  OPENANN_CHECK_EQUALS(layerState, NO_LAYER);
  OPENANN_CHECK_EQUALS(initializationState, UNINITIALIZED);
}

MLP::~MLP()
{
  if(optimizer)
    delete optimizer;
  if(deleteDataSetOnDestruction)
  {
    if(trainingData)
      delete trainingData;
    if(testData)
      delete testData;
  }
}

MLP& MLP::input(int units)
{
  OPENANN_CHECK_EQUALS(layerState, NO_LAYER);
  OPENANN_CHECK_EQUALS(initializationState, UNINITIALIZED);
  MLPImplementation::LayerInfo layerInfo(MLPImplementation::LayerInfo::INPUT, 1, units, MLPImplementation::ID);
  mlp.layerInfos.push_back(layerInfo);
  layerState = INPUT_LAYER;
  return *this;
}

MLP& MLP::input(int rows, int cols)
{
  OPENANN_CHECK_EQUALS(layerState, NO_LAYER);
  OPENANN_CHECK_EQUALS(initializationState, UNINITIALIZED);
  MLPImplementation::LayerInfo layerInfo(MLPImplementation::LayerInfo::INPUT, 2, rows*cols, MLPImplementation::ID);
  layerInfo.nodesPerDimension.push_back(rows);
  layerInfo.nodesPerDimension.push_back(cols);
  mlp.layerInfos.push_back(layerInfo);
  layerState = INPUT_LAYER;
  return *this;
}

MLP& MLP::input(int arrays, int rows, int cols)
{
  OPENANN_CHECK_EQUALS(layerState, NO_LAYER);
  OPENANN_CHECK_EQUALS(initializationState, UNINITIALIZED);
  MLPImplementation::LayerInfo layerInfo(MLPImplementation::LayerInfo::INPUT,
      3, arrays*rows*cols, MLPImplementation::ID);
  layerInfo.nodesPerDimension.push_back(arrays);
  layerInfo.nodesPerDimension.push_back(rows);
  layerInfo.nodesPerDimension.push_back(cols);
  mlp.layerInfos.push_back(layerInfo);
  layerState = INPUT_LAYER;
  return *this;
}

MLP& MLP::convolutionalLayer(int featureMaps, int kernelRows, int kernelCols, ActivationFunction a)
{
  OPENANN_CHECK(layerState == INPUT_LAYER || layerState == HIDDEN_LAYER_FINISHED);
  OPENANN_CHECK_EQUALS(initializationState, UNINITIALIZED);
  OPENANN_CHECK(a != SM);
  int lastLayerRows = mlp.layerInfos[mlp.layerInfos.size()-1].nodesPerDimension[1];
  int lastLayerCols = mlp.layerInfos[mlp.layerInfos.size()-1].nodesPerDimension[2];
  int featureMapRows = (lastLayerRows-kernelRows/2)/2;
  int featureMapCols = (lastLayerCols-kernelCols/2)/2;
  int nodes = featureMaps * featureMapRows * featureMapCols;
  MLPImplementation::LayerInfo layerInfo(MLPImplementation::LayerInfo::CONVOLUTIONAL,
      2, nodes, a == NO_ACT_FUN ? MLPImplementation::TANH :
      (MLPImplementation::ActivationFunction) a);
  layerInfo.nodesPerDimension.push_back(featureMaps);
  layerInfo.nodesPerDimension.push_back(featureMapRows);
  layerInfo.nodesPerDimension.push_back(featureMapCols);
  layerInfo.featureMaps = featureMaps;
  layerInfo.kernelRows = kernelRows;
  layerInfo.kernelCols = kernelCols;
  mlp.layerInfos.push_back(layerInfo);
  layerState = HIDDEN_LAYER_FINISHED;
  return *this;
}

MLP& MLP::fullyConnectedHiddenLayer(int units, ActivationFunction a, int numberOfParameters, std::string compression)
{
  OPENANN_CHECK(layerState == INPUT_LAYER || layerState == HIDDEN_LAYER_FINISHED);
  OPENANN_CHECK_EQUALS(initializationState, UNINITIALIZED);
  OPENANN_CHECK(a != SM);
  MLPImplementation::LayerInfo layerInfo(MLPImplementation::LayerInfo::FULLY_CONNECTED,
      1, units, a == NO_ACT_FUN ? MLPImplementation::TANH :
      (MLPImplementation::ActivationFunction) a);
  if(numberOfParameters > 0)
    layerInfo.compress(numberOfParameters, compression);
  mlp.layerInfos.push_back(layerInfo);
  layerState = HIDDEN_LAYER_FINISHED;
  return *this;
}

MLP& MLP::fullyConnectedHiddenLayer(int units, ActivationFunction a, int parameterRows, int parameterCols)
{
  fullyConnectedHiddenLayer(units, a, -1);
  compressWith(parameterRows, parameterCols);
  return *this;
}

MLP& MLP::fullyConnectedHiddenLayer(int units, ActivationFunction a, int parametersX, int parametersY, int parametersZ, const std::vector<std::vector<fpt> >& t)
{
  fullyConnectedHiddenLayer(units, a, -1);
  compressWith(parametersX, parametersY, parametersZ, t);
  return *this;
}

MLP& MLP::output(int units, ErrorFunction e, ActivationFunction a, int numberOfParameters, std::string compression)
{
  makeOutputLayer(units, e, a);
  if(numberOfParameters > 0)
    mlp.layerInfos[mlp.layerInfos.size()-1].compress(numberOfParameters, compression);
  finishInitialization();
  return *this;
}

MLP& MLP::output(int units, ErrorFunction e, ActivationFunction a, int parameterRows, int parameterCols)
{
  makeOutputLayer(units, e, a);
  compressWith(parameterRows, parameterCols);
  finishInitialization();
  return *this;
}

MLP& MLP::output(int units, ErrorFunction e, ActivationFunction a, int parametersX, int parametersY, int parametersZ, const std::vector<std::vector<fpt> >& t)
{
  makeOutputLayer(units, e, a);
  compressWith(parametersX, parametersY, parametersZ, t);
  finishInitialization();
  return *this;
}

MLP& MLP::makeOutputLayer(int units, ErrorFunction e, ActivationFunction a)
{
  OPENANN_CHECK(layerState == INPUT_LAYER || layerState == HIDDEN_LAYER_FINISHED);
  OPENANN_CHECK_EQUALS(initializationState, UNINITIALIZED);
  MLPImplementation::ActivationFunction g = (MLPImplementation::ActivationFunction) a;
  if(a == SM)
  {
    OPENANN_CHECK_EQUALS(CE, e);
    g = MLPImplementation::ID;
  }
  else if(a == NO_ACT_FUN)
    g = MLPImplementation::ID;
  MLPImplementation::LayerInfo layerInfo(MLPImplementation::LayerInfo::OUTPUT,
    1, units, g);
  mlp.layerInfos.push_back(layerInfo);
  layerState = OUTPUT_LAYER_FINISHED;
  errorFunction = e;
  return *this;
}

MLP& MLP::finishInitialization()
{
  mlp.init();
  mlp.initialize();
  initializationState = WEIGHTS_INIT;
  temporaryGradient.resize(dimension());
  temporaryOutput.resize(mlp.F);
  return *this;
}

MLP& MLP::compressWith(int parameterRows, int parameterCols)
{
  OPENANN_CHECK(layerState == HIDDEN_LAYER_FINISHED || layerState == OUTPUT_LAYER_FINISHED);
  OPENANN_CHECK_EQUALS(initializationState, UNINITIALIZED);
  OPENANN_CHECK_EQUALS(mlp.layerInfos[0].dimension, 2);
  mlp.parametersX = parameterRows;
  mlp.parametersY = parameterCols;
  mlp.layerInfos[mlp.layerInfos.size()-1].compress(parameterRows * parameterCols);
  return *this;
}

MLP& MLP::compressWith(int parametersX, int parametersY, int parametersZ, const std::vector<std::vector<fpt> >& t)
{
  OPENANN_CHECK(layerState == HIDDEN_LAYER_FINISHED || layerState == OUTPUT_LAYER_FINISHED);
  OPENANN_CHECK_EQUALS(initializationState, UNINITIALIZED);
  OPENANN_CHECK_EQUALS(mlp.layerInfos[0].dimension, 2);
  mlp.layerInfos[0].dimension = 3;
  mlp.parametersX = parametersX;
  mlp.parametersY = parametersY;
  mlp.parametersZ = parametersZ;
  mlp.layerInfos[mlp.layerInfos.size()-1].compress(parametersX * parametersY * parametersZ);
  mlp.firstLayer3Dt = t;
  return *this;
}

MLP& MLP::noBias()
{
  OPENANN_CHECK(initializationState < WEIGHTS_INIT);
  mlp.biased = false;
  return *this;
}

Learner& MLP::trainingSet(Mt& trainingInput, Mt& trainingOutput)
{
  if(deleteDataSetOnDestruction && trainingData)
    delete trainingData;
  deleteDataSetOnDestruction = true;
  trainingData = new DirectStorageDataSet(trainingInput, trainingOutput);
  N = trainingInput.cols();
  initializationState = INITIALIZED;
  return (Learner&) *this;
}

Learner& MLP::trainingSet(DataSet& trainingSet)
{
  if(deleteDataSetOnDestruction && trainingData)
    delete trainingData;
  deleteDataSetOnDestruction = false;
  trainingData = &trainingSet;
  N = trainingSet.samples();
  initializationState = INITIALIZED;
  return (Learner&) *this;
}

MLP& MLP::testSet(Mt& testInput, Mt& testOutput)
{
  if(deleteDataSetOnDestruction && testData)
    delete testData;
  deleteDataSetOnDestruction = true;
  testData = new DirectStorageDataSet(testInput, testOutput);
  return *this;
}

MLP& MLP::testSet(DataSet& testSet)
{
  if(deleteDataSetOnDestruction && testData)
    delete testData;
  deleteDataSetOnDestruction = false;
  testData = &testSet;
  return *this;
}

MLP& MLP::training(Training training, bool reinit)
{
  OPENANN_CHECK(initializationState >= UNINITIALIZED);
  reinitialize = reinit;
  if(optimizer)
  {
    delete optimizer;
    optimizer = 0;
  }
  switch(training)
  {
    case NOT_INITIALIZED:
      break;
    case BATCH_CMAES:
      optimizer = new IPOPCMAES;
      break;
    case BATCH_LMA:
#ifdef USE_GPL_LICENSE
      optimizer = new LMA(true);
      break;
#else
      errorLogger << "\n\nLMA is not available for LGPL license. Please compile"
            << " OpenANN with activated GPL license flag. See README.md for"
            << " instructions.\n\n";
      abort(1);
#endif
    case BATCH_SGD:
      optimizer = new SGD;
      break;
    default:
      OPENANN_CHECK(false && "unknown optimizer");
      break;
  }
  if(optimizer)
    optimizer->setOptimizable(*this);
  return *this;
}

Vt MLP::fit(StopCriteria stop)
{
  OPENANN_CHECK_EQUALS(initializationState, INITIALIZED);
  iteration = 0;

  std::stringstream stream;
  stream << "MLP ";
  for(size_t l = 0; l < mlp.layerInfos.size(); l++)
    stream << mlp.layerInfos[l].nodes << (l < mlp.layerInfos.size()-1 ? "-" : "");
  stream << " compressed with ";
  for(size_t l = 1; l < mlp.layerInfos.size(); l++)
    stream << mlp.layerInfos[l].parameters << (l < mlp.layerInfos.size()-1 ? "-" : "");
  configuration = stream.str();

  if(errorLogger.isActive())
    errorLogger << "\n# " << configuration << "\n"
                << "# Order of outputs:\n"
                << "# Training: SSE, correct, wrong, FP, TP, FN, TN\n"
                << "# (Test: SSE, correct, wrong, FP, TP, FN, TN)\n"
                << "# training time (ms)\n\n";
  if(parameterLogger.isActive())
    parameterLogger << "\n# " << configuration << "\n\n";

  optimizer->setStopCriteria(stop);
  sw.start();
  optimizer->optimize();
  return optimizer->result();
}

unsigned int MLP::dimension()
{
  OPENANN_CHECK(initializationState >= WEIGHTS_INIT);
  return (unsigned) mlp.P;
}

fpt MLP::error()
{
  OPENANN_CHECK_EQUALS(initializationState, INITIALIZED);
  fpt e = 0.0;
  for(int n = 0; n < N; n++)
    e += error(n);
  switch(errorFunction)
  {
    case SSE:
      return e / 2.0;
    case MSE:
      return e / (fpt) N;
    default:
      return e;
  }
}

bool MLP::providesGradient()
{
  OPENANN_CHECK_EQUALS(initializationState, INITIALIZED);
  return true;
}

Vt MLP::gradient()
{
  OPENANN_CHECK_EQUALS(initializationState, INITIALIZED);
  OPENANN_CHECK_EQUALS(temporaryGradient.rows(), dimension());
  temporaryGradient.fill(0.0);
  for(int n = 0; n < N; n++)
  {
    mlp(trainingData->getInstance(n));
    mlp.backpropagate(trainingData->getTarget(n));
    mlp.derivative(temporaryGradient);
  }
  switch(errorFunction)
  {
    case MSE:
      temporaryGradient /= (fpt) N;
    default:
      break;
  }
  return temporaryGradient;
}

bool MLP::providesHessian()
{
  OPENANN_CHECK_EQUALS(initializationState, INITIALIZED);
  return false;
}

Mt MLP::hessian()
{
  OPENANN_CHECK_EQUALS(initializationState, INITIALIZED);
  OPENANN_CHECK(false && "MLP does not provide a hessian matrix.");
  return Mt::Random(dimension(), dimension());
}

bool MLP::providesInitialization()
{
  OPENANN_CHECK(initializationState >= WEIGHTS_INIT);
  return true;
}

void MLP::initialize()
{
  OPENANN_CHECK(initializationState >= WEIGHTS_INIT);
  if(reinitialize)
    mlp.initialize();
}

void MLP::setParameters(const Vt& parameters)
{
  OPENANN_CHECK(initializationState >= WEIGHTS_INIT);
  mlp.set(parameters);
}

Vt MLP::currentParameters()
{
  OPENANN_CHECK(initializationState >= WEIGHTS_INIT);
  return mlp.get();
}

unsigned int MLP::examples()
{
  OPENANN_CHECK_EQUALS(initializationState, INITIALIZED);
  return N;
}

fpt MLP::error(unsigned int i)
{
  OPENANN_CHECK_EQUALS(initializationState, INITIALIZED);
  fpt e = 0.0;
  if(errorFunction == CE)
  {
    temporaryOutput = mlp(trainingData->getInstance(i));
    OpenANN::softmax(temporaryOutput);
    for(int f = 0; f < temporaryOutput.rows(); f++)
      e -= trainingData->getInstance(i)(f) * std::log(temporaryOutput(f));
  }
  else
  {
    temporaryOutput = mlp(trainingData->getInstance(i));
    const Vt diff = temporaryOutput - trainingData->getTarget(i);
    e += diff.dot(diff);
  }
  return e / 2.0;
}

Vt MLP::gradient(unsigned int i)
{
  OPENANN_CHECK_EQUALS(initializationState, INITIALIZED);
  OPENANN_CHECK_EQUALS(temporaryGradient.rows(), dimension());
  mlp(trainingData->getInstance(i));
  mlp.backpropagate(trainingData->getTarget(i));
  mlp.singleDerivative(temporaryGradient);
  return temporaryGradient;
}

Vt MLP::operator()(const Vt& x)
{
  OPENANN_CHECK(initializationState >= WEIGHTS_INIT);
  temporaryOutput = mlp(x);
  if(errorFunction == CE)
    OpenANN::softmax(temporaryOutput);
  return temporaryOutput;
}

int MLP::operator()(const Vt& x, Vt& fvec)
{
  setParameters(x);
  for(int n = 0; n < N; n++)
    fvec(n) = error(n);
  return 0;
}

int MLP::df(const Vt& x, Vt& fjac)
{
  setParameters(x);
  for(int n = 0; n < examples(); n++)
    fjac.row(n) = gradient(n);
  return 0;
}

void MLP::VJ(Vt& values, Mt& jacobian)
{
  OPENANN_CHECK_EQUALS(initializationState, INITIALIZED);
  OPENANN_CHECK_EQUALS(values.rows(), (int) examples());
  OPENANN_CHECK_EQUALS(jacobian.rows(), (int) examples());
  OPENANN_CHECK_EQUALS(jacobian.cols(), (int) dimension());
  for(unsigned n = 0; n < examples(); n++)
  {
    temporaryOutput = mlp(trainingData->getInstance(n)) - trainingData->getTarget(n);
    Eigen::Matrix<fpt, 1, 1> e = temporaryOutput.transpose() * temporaryOutput / (fpt) 2.0;
    values(n) = e(0,0);
    mlp.backpropagate(trainingData->getTarget(n));
    mlp.singleDerivative(temporaryGradient);
    jacobian.row(n) = temporaryGradient;
  }
}

void MLP::finishedIteration()
{
  if(errorLogger.isActive())
  {
    fpt trainingError = 0.0;
    fpt testError = 0.0;
    int tefp = 0, tetp = 0, tefn = 0, tetn = 0;
    int trfp = 0, trtp = 0, trfn = 0, trtn = 0;
    switch(mlp.F)
    {
      case 1:
      {
        if(trainingData)
        {
          for(int i = 0; i < trainingData->samples(); i++)
          {
            const Vt yv = (*this)(trainingData->getInstance(i));
            const fpt y = yv(0, 0);
            const Vt tv = trainingData->getTarget(i);
            const fpt t = tv(0, 0);
            trainingError += std::pow(y-t, 2.0);
            if(y < 0 && t < 0)
              trtn++;
            else if(y < 0 && t >= 0)
              trfn++;
            else if(y >= 0 && t < 0)
              trfp++;
            else
              trtp++;
          }
          trainingError /= 2.0;
        }
        if(testData)
        {
          for(int i = 0; i < testData->samples(); i++)
          {
            const Vt yv = (*this)(testData->getInstance(i));
            const fpt y = yv(0, 0);
            const Vt tv = testData->getTarget(i);
            const fpt t = tv(0, 0);
            testError += std::pow(y-t, 2.0);
            if(y < 0 && t < 0)
              tetn++;
            else if(y < 0 && t >= 0)
              tefn++;
            else if(y >= 0 && t < 0)
              tefp++;
            else
              tetp++;
          }
          testError /= 2.0;
        }
      }
      break;
      default:
      {
        if(trainingData)
        {
          for(int i = 0; i < trainingData->samples(); i++)
          {
            const Vt y = (*this)(trainingData->getInstance(i));
            const Vt t = trainingData->getTarget(i);
            int yClass = -1;
            int tClass = -1;
            for(int f = 0; f < t.rows(); f++)
              trainingError -= t(f) * std::log(y(f));
            y.maxCoeff(&yClass);
            t.maxCoeff(&tClass);
            if(yClass == tClass)
              trtp++;
            else
              trfp++;
          }
        }
        if(testData)
        {
          for(int i = 0; i < testData->samples(); i++)
          {
            const Vt y = (*this)(testData->getInstance(i));
            const Vt t = testData->getTarget(i);
            int yClass = -1;
            int tClass = -1;
            for(int f = 0; f < t.rows(); f++)
              testError -= t(f) * std::log(y(f));
            y.maxCoeff(&yClass);
            t.maxCoeff(&tClass);
            if(yClass == tClass)
              tetp++;
            else
              tefp++;
          }
        }
      }
      break;
    }
    const int correctTrain = trtn + trtp;
    const int wrongTrain = trfn + trfp;
    const int correctTest = tetn + tetp;
    const int wrongTest = tefn + tefp;

    errorLogger << ++iteration << " ";
    if(trainingData)
    {
      errorLogger << trainingError << " " << correctTrain << " " << wrongTrain
          << " " << trfp << " " << trtp << " " << trfn << " " << trtn << " ";
    }
    if(testData)
    {
      errorLogger << testError << " " << correctTest << " " << wrongTest << " "
          << tefp << " " << tetp << " " << tefn << " " << tetn;
    }
    errorLogger << " " << sw.stop(Stopwatch::MILLISECOND) << "\n";
  }

  if(parameterLogger.isActive())
    parameterLogger << iteration << "\n" << mlp.get().transpose() << "\n";

  if(trainingData)
    trainingData->finishIteration(*this);
  if(testData)
    testData->finishIteration(*this);
}

}
