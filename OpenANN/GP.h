#pragma once

#include <Learner.h>
#include <Optimizable.h>

namespace OpenANN {

class GP : public Learner, public Optimizable
{
private:
  Vt parameters;
  fpt& beta;
  fpt& theta0;
  fpt& theta1;
  fpt& theta2;
  fpt& theta3;
  DataSet* dataSet;
  bool deleteDataSetOnDestruction;

  Mt covarianceInv;
  Mt t;
  fpt var;
public:
  GP(fpt beta, fpt theta0, fpt theta1, fpt theta2, fpt theta3);
  ~GP();
  virtual Vt currentParameters();
  virtual unsigned int dimension();
  virtual fpt error();
  virtual Vt gradient();
  virtual Mt hessian();
  virtual void initialize();
  virtual bool providesGradient();
  virtual bool providesHessian();
  virtual bool providesInitialization();
  virtual void setParameters(const Vt& parameters);
  virtual Learner& trainingSet(Mt& trainingInput, Mt& trainingOutput);
  virtual Learner& trainingSet(DataSet& trainingSet);
  void buildModel();
  virtual Vt operator()(const Vt& x);
  fpt variance();

private:
  fpt kernel(const Vt& x1, const Vt& x2);
};

}
