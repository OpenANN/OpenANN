#ifndef OPENANN_RBM_H_
#define OPENANN_RBM_H_

#include <OpenANN/Learner.h>
#include <OpenANN/Regularization.h>
#include <OpenANN/layers/Layer.h>
#include <OpenANN/optimization/Optimizable.h>
#include <OpenANN/optimization/StoppingCriteria.h>
#include <OpenANN/util/Random.h>
#include <Eigen/Dense>
#include <vector>

namespace OpenANN
{

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
 * Supports the following regularization types:
 *
 * - L1 penalty
 * - L2 penalty
 *
 * [1] Smolensky, Paul:
 * Information Processing in Dynamical Systems: Foundations of Harmony Theory,
 * MIT Press, 1986, pp. 194-281.
 *
 * [2] Hinton, Geoffrey E.:
 * Training Products of Experts by Minimizing Contrastive Divergence,
 * Technical Report, University College London, 2000.
 */
class RBM : public Learner, public Layer
{
  RandomNumberGenerator rng;
  int D, H;
  int cdN;
  double stdDev;
  Eigen::MatrixXd W, posGradW, negGradW, Wd;
  Eigen::VectorXd bv, posGradBv, negGradBv, bh, posGradBh, negGradBh, bhd;
  Eigen::MatrixXd pv, v, ph, h, phd;
  Eigen::MatrixXd deltas, e;
  int K;
  Eigen::VectorXd params, grad;
  bool backprop;
  Regularization regularization;

public:
  /**
   * Construct RBM.
   *
   * @param D number of inputs
   * @param H number of hidden nodes
   * @param cdN number of contrastive divergence steps
   * @param stdDev standard deviation of initial weights
   * @param backprop weights can be finetuned with backprop
   * @param regularization regularization coefficients
   */
  RBM(int D, int H, int cdN = 1, double stdDev = 0.01, bool backprop = true,
      Regularization regularization = Regularization());
  virtual ~RBM() {}

  // Learner interface
  virtual Eigen::VectorXd operator()(const Eigen::VectorXd& x);
  virtual Eigen::MatrixXd operator()(const Eigen::MatrixXd& X);
  virtual bool providesInitialization();
  virtual void initialize();
  virtual unsigned int examples();
  virtual unsigned int dimension();
  virtual void setParameters(const Eigen::VectorXd& parameters);
  virtual const Eigen::VectorXd& currentParameters();
  virtual double error();
  virtual double error(unsigned int n);
  virtual bool providesGradient();
  virtual Eigen::VectorXd gradient();
  virtual Eigen::VectorXd gradient(unsigned int n);
  virtual void errorGradient(std::vector<int>::const_iterator startN,
                             std::vector<int>::const_iterator endN,
                             double& value, Eigen::VectorXd& grad);

  // Layer interface
  virtual void backpropagate(Eigen::MatrixXd* ein, Eigen::MatrixXd*& eout,
                             bool backpropToPrevious);
  virtual void forwardPropagate(Eigen::MatrixXd* x, Eigen::MatrixXd*& y,
                                bool dropout);
  virtual Eigen::MatrixXd& getOutput();
  virtual OutputInfo initialize(std::vector<double*>& parameterPointers,
                                std::vector<double*>& parameterDerivativePointers);
  virtual void initializeParameters() {}
  virtual void updatedParameters() {}
  virtual Eigen::VectorXd getParameters();

  // RBM interface
  int visibleUnits();
  int hiddenUnits();
  const Eigen::MatrixXd& getWeights();
  const Eigen::MatrixXd& getVisibleProbs();
  const Eigen::MatrixXd& getVisibleSample();
  Eigen::MatrixXd reconstructProb(int n, int steps);
  void sampleHgivenV();
  void sampleVgivenH();
private:
  void reality();
  void daydream();
  void fillGradient();
};

} // namespace OpenANN

#endif // OPENANN_RBM_H_
