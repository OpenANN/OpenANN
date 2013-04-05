#pragma once

#include <optimization/Optimizable.h>
#include <Learner.h>
#include <optimization/StoppingCriteria.h>
#include <Eigen/Dense>

namespace OpenANN {

class RBM : public Optimizable, public Learner
{
public:
  int D, H;
  int cdN;
  fpt stdDev;
  Mt W, posGradW, negGradW;
  Vt bv, posGradBv, negGradBv, bh, posGradBh, negGradBh;
  Vt pv, v, ph, h;
  DataSet* trainSet; // TODO unify code to store data sets (move to learner)

  RBM(int D, int H, int cdN = 1, fpt stdDev = 0.01);
  virtual Vt operator()(const Vt& x);
  virtual bool providesInitialization();
  virtual void initialize();
  virtual unsigned int examples();
  virtual unsigned int dimension();
  virtual void setParameters(const Vt& parameters);
  virtual Vt currentParameters();
  virtual fpt error();
  virtual bool providesGradient();
  virtual Vt gradient();
  virtual Vt gradient(unsigned int i);
  virtual bool providesHessian();
  virtual Mt hessian();
  virtual Learner& trainingSet(Mt& trainingInput, Mt& trainingOutput);
  virtual Learner& trainingSet(DataSet& trainingSet);

private:
  void reality(int n);
  void daydream();
  void sampleHgivenV();
  void sampleVgivenH();
};

}
