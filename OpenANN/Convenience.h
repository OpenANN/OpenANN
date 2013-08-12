#ifndef OPENANN_CONVENIENCE_H_
#define OPENANN_CONVENIENCE_H_

#include <OpenANN/Net.h>
#include <string>

namespace OpenANN
{


class StoppingCriteria;


/**
 * Train a feedforward neural network supervised.
 *
 * @param net neural network
 * @param algorithm a registered algorithm, e.g. "MBSGD", "LMA", "CG", "LBFGS"
 *                  or "CMAES"
 * @param errorFunction error function to optimize
 * @param stop stopping criteria
 * @param reinitialize should the weights be initialized before optimization?
 * @param dropout use dropout for regularization
 */
void train(Net& net, std::string algorithm, ErrorFunction errorFunction,
           const StoppingCriteria& stop, bool reinitialize = false,
           bool dropout = false);

/**
 * Create a multilayer neural network.
 *
 * @param net neural network
 * @param g activation function in hidden layers
 * @param h activation function in output layer
 * @param D number of inputs
 * @param F number of outputs
 * @param H number of hidden layers
 * @param ... numbers of hidden units
 */
void makeMLNN(Net& net, ActivationFunction g, ActivationFunction h,
              int D, int F, int H, ...);

} // namespace OpenANN

#endif // OPENANN_CONVENIENCE_H_
