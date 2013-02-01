#pragma once

#include <Optimizable.h>
#include <Learner.h>
#include <layers/Layer.h>
#include <ActivationFunctions.h>
#include <StopCriteria.h>
#include <io/Logger.h>
#include <vector>

namespace OpenANN {

class DeepNetwork : public Optimizable, Learner
{
public:
  enum ErrorFunction
  {
    NO_E_DEFINED,
    SSE, //!< Sum of squared errors and identity (regression)
    MSE, //!< Mean of squared errors and identity (regression)
    CE   //!< Cross entropy and softmax (classification)
  };

  enum Training
  {
    NOT_INITIALIZED,
    BATCH_CMAES,  //!< Covariance Matrix Adaption Evolution Strategies
    BATCH_LMA,    //!< Levenberg-Marquardt Algorithm
    BATCH_SGD     //!< Stochastic Gradient Descent
  };

private:
  Logger debugLogger;
  std::vector<OutputInfo> infos;
  std::vector<Layer*> layers;
  std::vector<fpt*> parameters;
  std::vector<fpt*> derivatives;
  DataSet* dataSet;
  DataSet* testDataSet;
  bool deleteDataSet, deleteTestSet;
  ErrorFunction errorFunction;

  bool initialized;
  int P, N, L;
  Vt parameterVector;
  Vt tempInput, tempOutput, tempError, tempGradient;

public:
  DeepNetwork(ErrorFunction errorFunction);
  virtual ~DeepNetwork();
  DeepNetwork& inputLayer(int dim1, int dim2 = 1, int dim3 = 1, bool bias = true);
  DeepNetwork& fullyConnectedLayer(int units,
                                   ActivationFunction act,
                                   fpt stdDev = (fpt) 0.5, bool bias = true);
  DeepNetwork& convolutionalLayer(int featureMaps, int kernelRows,
                                  int kernelCols, ActivationFunction act,
                                  fpt stdDev = (fpt) 0.5, bool bias = true);
  DeepNetwork& subsamplingLayer(int kernelRows, int kernelCols,
                                ActivationFunction act,
                                fpt stdDev = (fpt) 0.5, bool bias = true);
  DeepNetwork& outputLayer(int units, ActivationFunction act,
                           fpt stdDev = (fpt) 0.5);

  virtual Learner& trainingSet(Mt& trainingInput, Mt& trainingOutput);
  virtual Learner& trainingSet(DataSet& trainingSet);
  virtual DeepNetwork& testSet(Mt& testInput, Mt& testOutput);
  virtual DeepNetwork& testSet(DataSet& testDataSet);
  Vt train(Training algorithm, StopCriteria stop, bool reinitialize = true);
  virtual void finishedIteration();

  virtual Vt operator()(const Vt& x);
  virtual unsigned int dimension();
  virtual unsigned int examples();
  virtual Vt currentParameters();
  virtual void setParameters(const Vt& parameters);
  virtual bool providesInitialization();
  virtual void initialize();
  virtual fpt error(unsigned int i);
  virtual fpt error();
  virtual bool providesGradient();
  virtual Vt gradient(unsigned int i);
  virtual Vt gradient();
  virtual void VJ(Vt& values, Mt& jacobian);
  virtual bool providesHessian();
  virtual Mt hessian();
};

}
