#ifndef OPENANN_CONVENIENCE_H_
#define OPENANN_CONVENIENCE_H_

#include <OpenANN/Net.h>
#include <string>

namespace OpenANN
{

/**
 * Train a feedforward neural network supervised.
 *
 * @param net neural network
 * @param algorithm a registered algorithm, e.g. "MBSGD", "LMA", "CG" or
 *                  "CMAES"
 * @param errorFunction error function to optimize
 * @param stop stopping criteria
 * @param reinitialize should the weights be initialized before optimization?
 * @param dropout use dropout for regularization
 */
void train(Net& net, std::string algorithm, ErrorFunction errorFunction,
           StoppingCriteria stop, bool reinitialize = false,
           bool dropout = false);

}

#endif // OPENANN_CONVENIENCE_H_
