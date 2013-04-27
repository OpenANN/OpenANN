#pragma once

#include <OpenANN/Learner.h>
#include <OpenANN/optimization/Optimizable.h>
#include <OpenANN/optimization/StoppingCriteria.h>
#include <OpenANN/util/Random.h>
#include <Eigen/Dense>

namespace OpenANN {

class RBM : public Learner
{
public:
  RandomNumberGenerator rng;
  int D, H;
  int cdN;
  double stdDev;
  Eigen::MatrixXd W, posGradW, negGradW;
  Eigen::VectorXd bv, posGradBv, negGradBv, bh, posGradBh, negGradBh;
  Eigen::VectorXd pv, v, ph, h;
  DataSet* trainSet; // TODO unify code to store data sets (move to learner)

  RBM(int D, int H, int cdN = 1, double stdDev = 0.01);
  virtual Eigen::VectorXd operator()(const Eigen::VectorXd& x);
  virtual bool providesInitialization();
  virtual void initialize();
  virtual unsigned int examples();
  virtual unsigned int dimension();
  virtual void setParameters(const Eigen::VectorXd& parameters);
  virtual Eigen::VectorXd currentParameters();
  virtual double error();
  virtual bool providesGradient();
  virtual Eigen::VectorXd gradient();
  virtual Eigen::VectorXd gradient(unsigned int i);
  virtual bool providesHessian();
  virtual Eigen::MatrixXd hessian();
  virtual Learner& trainingSet(Eigen::MatrixXd& trainingInput,
                               Eigen::MatrixXd& trainingOutput);
  virtual Learner& trainingSet(DataSet& trainingSet);

  Eigen::VectorXd reconstructProb(int n, int steps);
  Eigen::VectorXd reconstruct(int n, int steps);
private:
  void reality(int n);
  void daydream();
  void sampleHgivenV();
  void sampleVgivenH();
};

}
