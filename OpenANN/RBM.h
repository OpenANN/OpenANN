#pragma once

#include <OpenANN/Learner.h>
#include <OpenANN/optimization/Optimizable.h>
#include <OpenANN/optimization/StoppingCriteria.h>
#include <OpenANN/util/Random.h>
#include <Eigen/Dense>

namespace OpenANN {

/**
 * @class RBM
 *
 * Restricted Boltzmann Machine.
 *
 * RBMs have been originally invented by Paul Smolensky in 1986 [1] and since
 * contrastive divergence [2] can be used to calculate an approximation of a
 * gradient, we can efficiently train RBMs. RBMs are usually used to learn
 * features unsupervised. However, they can be stacked and we can use them
 * to initialize deep autoencoders for dimensionality reduction or feedforward
 * networks for classification. Standard RBMs assume that the data is binary
 * (at least approximately, i.e. the values have to be within [0, 1]).
 *
 * Deep networks are usually difficult to train because the required learning
 * rate in the first layer is usually much higher than in the upper layers.
 * This problem can be solved by initializing the first layers with RBMs,
 * which was the major breakthrouh in deep learning. There are also other ways
 * to make deep learning work, e.g. CNNs (weight sharing), ReLUs, maxout, etc.
 *
 * [1] Smolensky, Paul:
 * Information Processing in Dynamical Systems: Foundations of Harmony Theory,
 * MIT Press, 1986, pp. 194-281.
 *
 * [2] Hinton, Geoffrey E.:
 * Training Products of Experts by Minimizing Contrastive Divergence,
 * Technical Report, University College London, 2000.
 */
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
  int K;
  Eigen::VectorXd params;
  DataSet* trainSet;

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

  void reality(int n);
  void daydream();
  void sampleHgivenV();
  void sampleVgivenH();
};

}
