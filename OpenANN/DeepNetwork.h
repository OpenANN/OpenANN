#pragma once

#include <Optimizable.h>
#include <Learner.h>
#include <layers/Layer.h>
#include <ActivationFunctions.h>
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

private:
  std::vector<OutputInfo> infos;
  std::vector<Layer*> layers;
  std::list<fpt*> parameters;
  std::list<fpt*> derivatives;
  DataSet* dataSet;
  bool deleteDataSet;
  ErrorFunction errorFunction;

public:
  DeepNetwork(ErrorFunction errorFunction);
  virtual ~DeepNetwork();
  DeepNetwork& inputLayer(bool bias, int dim1, int dim2 = 1, int dim3 = 1);
  DeepNetwork& fullyConnectedLayer(bool bias, int units,
                                   ActivationFunction act,
                                   fpt stdDev = (fpt) 0.05);
  DeepNetwork& convolutionalLayer(bool bias, int featureMaps, int kernelRows,
                                  int kernelCols, ActivationFunction act,
                                  fpt stdDev = (fpt) 0.05);
  DeepNetwork& subsamplingLayer(bool bias, int kernelRows, int kernelCols,
                                ActivationFunction act,
                                fpt stdDev = (fpt) 0.05);

  virtual Learner& trainingSet(Mt& trainingInput, Mt& trainingOutput);
  virtual Learner& trainingSet(DataSet& trainingSet);
  virtual Vt operator()(const Vt& x);
  virtual unsigned int dimension();
  virtual Vt currentParameters();
  virtual void setParameters(const Vt& parameters);
  virtual bool providesInitialization();
  virtual void initialize();
  virtual fpt error(unsigned int i);
  virtual fpt error();
  virtual bool providesGradient();
  virtual Vt gradient(unsigned int i);
  virtual Vt gradient();
  virtual bool providesHessian();
  virtual Mt hessian();
};

}
