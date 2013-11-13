#ifndef OPENANN_OPTIMIZATION_MBSGD_H_
#define OPENANN_OPTIMIZATION_MBSGD_H_

#include <OpenANN/optimization/Optimizer.h>
#include <OpenANN/optimization/StoppingCriteria.h>
#include <OpenANN/util/Random.h>
#include <Eigen/Core>
#include <vector>
#include <list>

namespace OpenANN
{

/**
 * @class MBSGD
 *
 * Mini-batch stochastic gradient descent.
 *
 * This implementation of gradient descent has some modifications:
 *
 * - it is stochastic, we update the weights with a randomly chosen subset of
 *   the training set to escape local minima and thus increase the
 *   generalization capabilities
 * - we use a momentum to smooth the search direction
 * - each weight has an adaptive learning rate
 * - we can decrease the learning rate during optimization
 * - we can increase the momentum during optimization
 *
 * A good introduction to optimization with MBSGD can be found in Geoff
 * Hinton's Coursera course "Neural Networks for Machine Learning". A detailed
 * description of this implementation follows.
 *
 * When the batch size equals 1, the algorithms degenerates to stochastic
 * gradient descent. When it equals the training set size, the algorithm is
 * like batch gradient descent.
 *
 * Standard mini-batch stochastic gradient descent updates the weight vector w
 * in step t through
 *
 * \f$ w^t = w^{t-1} - \frac{\alpha}{|B_t|} \sum_{n \in B_t} \nabla E_n(w) \f$
 *
 * or
 *
 * \f$ \Delta w^t = - \frac{\alpha}{|B_t|} \sum_{n \in B_t} \nabla E_n(w),
 *     \quad w^t = w^{t-1} + \Delta w^t, \f$
 *
 * where \f$ \alpha \f$ is the learning rate and \f$ B_t \f$ is the set of
 * indices of the t-th mini-batch, which is drawn randomly. The random order
 * of gradients prevents us from getting stuck in local minima. This is an
 * advantage over batch gradient descent. However, we must not make the batch
 * size too small. A bigger batch size makes the optimization more robust
 * against noise of the training set. A reasonable batch size is between 10
 * and 100. The learning rate has to be within [0, 1). High learning rates can
 * result in divergence, i.e. the error increases. Too low learning rates
 * might make learning too slow, i.e. the number of epochs required to find an
 * optimum might be infeasibe. A reasonable value for :math:`\\alpha` is
 * usually within [1e-5, 0.1].
 *
 * A momentum can increase the optimization stability. In this case, the
 * update rule is
 *
 * \f$ \Delta w^t = \eta \Delta w^{t-1} - \frac{\alpha}{|B_t|}
 *                  \sum_{n \in B_t} \nabla E_n(w), \quad w^t = w^{t-1} +
 *                  \Delta w^t, \f$
 *
 * where \f$ \eta \f$ is called momentum and must lie in [0, 1). The momentum
 * term incorporates past gradients with exponentially decaying influence.
 * This reduces changes of the search direction. An intuitive explanation of
 * this update rule is: we regard w as the position of a ball that is rolling
 * down a hill. The gradient represents its acceleration and the acceleration
 * modifies its momentum.
 *
 * Another type of momentum that is available in this implementation is the
 * Nesterov's accelerated gradient (NAG). For smooth convex functions, NAG
 * achieves a convergence rate of \f$ O(\frac{1}{T^2}) \f$ instead of
 * \f$ O(\frac{1}{T}) \f$ [1]. The update rule only differs in where we
 * calculate the gradient.
 *
 * \f$ \Delta w^t = \eta \Delta w^{t-1} - \frac{\alpha}{|B_t|}
 *                  \sum_{n \in B_t} \nabla E_n(w + \eta \Delta w^{t-1}),
 *                  \quad w^t = w^{t-1} + \Delta w^t, \f$
 *
 * Another trick is using different learning rates for each weight. For each
 * weight \f$ w_{ji} \f$ we can introduce a gain \f$ g_{ji} \f$ which will be
 * multiplied with the learning rate so that we obtain an update rule for each
 * weight
 *
 * \f$ \Delta w_{ji}^t = \eta \Delta w_{ji}^{t-1} -
 *     \frac{\alpha g_{ji}^{t-1}}{|B_t|} \sum_{n \in B_t} \nabla E_n(w_{ji}),
 *     \quad w_{ji}^t = w_{ji}^{t-1} + \Delta w_{ji}^t, \f$
 *
 * where \f$ g_{ji}^0 = 1 \f$ and \f$ g_{ji} \f$ will be increased by 0.05 if
 * \f$ \Delta w_{ji}^t \Delta w_{ji}^{t-1} \geq 0 \f$, i.e. the sign of the
 * search direction did not change and \f$ g_{ji} \f$ will be multiplied by
 * 0.95 otherwise. We set a minimum and a maximum value for each gain. Usually
 * these are 0.1 and 10 or 0.001 and 100 respectively.
 *
 * During optimization it often makes sense to start with a more global
 * search, i.e. with a high learning rate and decrease the learning rate as we
 * approach the minimum so that we obtain an update rule for the learning
 * rate:
 *
 * \f$ \alpha^t = max(\alpha_{decay} \alpha^{t-1}, \alpha_{min}). \f$
 *
 * In addition, we can allow the optimizer to change the search direction more
 * often at the beginning of the optimization and reduce this possibility at
 * the end. To do this, we can start with a low momentum and increase it over
 * time until we reach a maximum:
 *
 * \f$ \eta^t = min(\eta^{t-1} + \eta_{inc}, \eta_{max}). \f$
 *
 * [1] Sutskever, Ilya; Martens, James; Dahl, George; Hinton, Geoffrey:
 * On the importance of initialization and momentum in deep learning,
 * International Conference on Machine Learning, 2013.
 */
class MBSGD : public Optimizer
{
  //! Stopping criteria
  StoppingCriteria stop;
  //! Optimizable problem
  Optimizable* opt; // do not delete
  //! Use nesterov's accelerated momentum
  bool nesterov;
  //! Learning rate
  double alpha;
  //! Learning rate decay
  double alphaDecay;
  //! Minimum learning rat
  double minAlpha;
  //! Momentum
  double eta;
  //! Momentum gain
  double etaGain;
  //! Maximum momentum
  double maxEta;
  //! Typical size of a mini-batch is 10 to a few hundred.
  int batchSize;
  //! Minimum gain of the learning rate per parameter, e. g. 0.01, 0.1 or 1
  double minGain;
  //! Maximum gain of the learning rate per parameter, e. g. 1, 10 or 100
  double maxGain;
  //! Use parameter adaption
  bool useGain;

  int iteration;
  RandomNumberGenerator rng;
  int P, N, batches;
  Eigen::VectorXd gradient, gains, parameters, momentum, currentGradient;
  double accumulatedError;
  std::vector<int> randomIndices;
public:
  /**
   * Create mini-batch stochastic gradient descent optimizer.
   *
   * @param learningRate learning rate (usually called alpha); range: (0, 1]
   * @param momentum momentum coefficient (usually called eta); range: [0, 1)
   * @param batchSize size of the mini-batches; range: [1, N], where N is the
   *                  size of the training set
   * @param nesterov use nesterov's accelerated momentum
   * @param learningRateDecay will be multiplied with the learning rate after
   *                          each weight update; range: (0, 1]
   * @param minimalLearningRate minimum value for the learning rate; range:
   *                            [0, 1]
   * @param momentumGain will be added to the momentum after each weight
   *                     update; range: [0, 1)
   * @param maximalMomentum maximum value for the momentum; range [0, 1]
   * @param minGain minimum factor for individual learning rates
   * @param maxGain maximum factor for individual learning rates
   */
  MBSGD(double learningRate = 0.01, double momentum = 0.5, int batchSize = 10,
        bool nesterov = false, double learningRateDecay = 1.0,
        double minimalLearningRate = 0.0, double momentumGain = 0.0,
        double maximalMomentum = 1.0, double minGain = 1.0,
        double maxGain = 1.0);
  ~MBSGD();
  virtual void setOptimizable(Optimizable& opt);
  virtual void setStopCriteria(const StoppingCriteria& stop);
  virtual void optimize();
  virtual bool step();
  virtual Eigen::VectorXd result();
  virtual std::string name();
private:
  void initialize();
};

} // namespace OpenANN

#endif // OPENANN_OPTIMIZATION_MBSGD_H_
