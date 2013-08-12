#include <OpenANN/Convenience.h>
#include <OpenANN/util/OpenANNException.h>
#include <OpenANN/optimization/Optimizer.h>
#include <OpenANN/optimization/MBSGD.h>
#include <OpenANN/optimization/LMA.h>
#include <OpenANN/optimization/CG.h>
#include <OpenANN/optimization/LBFGS.h>
#include <OpenANN/optimization/IPOPCMAES.h>
#include <OpenANN/optimization/StoppingCriteria.h>
#include <cstdarg>

namespace OpenANN
{

void train(Net& net, std::string algorithm, ErrorFunction errorFunction,
           const StoppingCriteria& stop, bool reinitialize, bool dropout)
{
  if(reinitialize)
    net.initialize();
  net.setErrorFunction(errorFunction);
  net.useDropout(dropout);

  Optimizer* opt;
  if(algorithm == "MBSGD")
    opt = new MBSGD;
  else if(algorithm == "LMA")
    opt = new LMA;
  else if(algorithm == "CG")
    opt = new CG;
  else if(algorithm == "LBFGS")
    opt = new LBFGS();
  else if(algorithm == "CMAES")
    opt = new IPOPCMAES;
  else
    throw OpenANNException("Unknown optimizer: " + algorithm);

  opt->setOptimizable(net);
  opt->setStopCriteria(stop);
  opt->optimize();
  opt->result();

  delete opt;
  net.useDropout(false);
}

void makeMLNN(Net& net, ActivationFunction g, ActivationFunction h,
              int D, int F, int H, ...)
{
  std::va_list nodes;
  va_start(nodes, H);

  net.inputLayer(D);
  for(int i = 0; i < H; i++)
  {
    const int j = va_arg(nodes, int);
    net.fullyConnectedLayer(j, g);
  }
  net.outputLayer(F, h);
}

} // namespace OpenANN
