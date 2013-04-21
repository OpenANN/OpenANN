#include <OpenANN/Convenience.h>
#include <OpenANN/util/OpenANNException.h>
#include <OpenANN/optimization/Optimizer.h>
#include <OpenANN/optimization/MBSGD.h>
#include <OpenANN/optimization/LMA.h>
#include <OpenANN/optimization/IPOPCMAES.h>

namespace OpenANN {

void train(Net& net, std::string algorithm, ErrorFunction errorFunction,
           StoppingCriteria stop, bool reinitialize, bool dropout)
{
  if(reinitialize)
    net.initialize();
  net.setErrorFunction(errorFunction);
  net.useDropout(dropout);

  Optimizer* opt;
  if(algorithm == "MBSGD")
    opt = new MBSGD;
  else if(algorithm == "CMAES")
    opt = new IPOPCMAES;
  else if(algorithm == "LMA")
    opt = new LMA;
  else
    throw OpenANNException("Unknown optimizer: " + algorithm);

  opt->setOptimizable(net);
  opt->setStopCriteria(stop);
  opt->optimize();
  opt->result();

  delete opt;
  net.useDropout(false);
}

}
