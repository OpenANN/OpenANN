#ifndef OPENANN_REGULARIZATION_H_
#define OPENANN_REGULARIZATION_H_

namespace OpenANN
{

/**
 * @class Regularization
 *
 * Holds all information related to regularization terms in an error function.
 *
 * Usually we do not penalize biases.
 */
class Regularization
{
public:
  //! Penalty for absolute weights, encourages sparse weight matrices.
  double l1Penalty;
  //! Penalty for squared weights.
  double l2Penalty;
  //! Maximum value for squared norm of the weight vector of a single neuron.
  double maxSquaredWeightNorm;

  Regularization(double l1Penalty = 0.0, double l2Penalty = 0.0,
                 double maxSquaredWeightNorm = 0.0)
    : l1Penalty(l1Penalty), l2Penalty(l2Penalty),
      maxSquaredWeightNorm(maxSquaredWeightNorm)
  {
  }
};

} // namespace OpenANN

#endif // OPENANN_REGULARIZATION_H_
