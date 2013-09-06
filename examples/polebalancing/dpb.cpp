#include <iostream>
#include <QApplication>
#include <OpenANN/OpenANN>
#include "DoublePoleBalancingVisualization.h"

/**
 * \page PB Pole Balancing
 *
 * Let a neural network learn how to balance poles that are mounted on a cart.
 *
 * This is an example for a reinforcement learning problem. We will use direct
 * policy search, i.e. we will not approximate a value function from which we
 * will infer a policy. Instead, we will represent the policy with a neural
 * network and we will optimize its parameters with a gradient-free optimization
 * algorithm (CMAES).
 *
 * You can choose between the environments SinglePoleBalancing and
 * DoublePoleBalancing with or without velocities. The NeuroEvolutionAgent
 * solves this problem.
 *
 * Usage:
 *
 * Available command line arguments are:
 * - "-spb": Single Pole Balancing
 * - "-po": Partially Observable (without velocities)
 * - "-ab": Use alpha beta filters to estimate the velocities
 * - "-des": Use double exponential smoothing to estimate the velocities
 *
 * Start the experiment with the key "r". Increase the simulation speed with
 * "+" and decrease the simulation speed with "-".
 *
 * \image html polebalancing.png
 */

void printUsage()
{
  std::cout << "unrecognized option, usage:" << std::endl
            << "\t-spb\tSingle Pole Balancing" << std::endl
            << "\t-po\tPartially Observable (without velocities)" << std::endl
            << "\t-ab\tUse alpha beta filters to estimate the velocities" << std::endl
            << "\t-des\tUse double exponential smoothing to estimate the velocities" << std::endl;
}

int main(int argc, char** argv)
{
  OpenANN::OpenANNLibraryInfo::print();

  bool singlePoleBalancing = false;
  bool partiallyObservable = false;
  bool useAlphaBetaFilters = false;
  bool useDoubleExponentialSmoothing = false;
  for(int i = 1; i < argc; i++)
  {
    std::string argument(argv[i]);
    if(argument == std::string("-spb"))
      singlePoleBalancing = true;
    else if(argument == std::string("-po"))
      partiallyObservable = true;
    else if(argument == std::string("-ab"))
      useAlphaBetaFilters = true;
    else if(argument == std::string("-des"))
      useDoubleExponentialSmoothing = true;
    else
      printUsage();
  }

  QApplication app(argc, argv);
  DoublePoleBalancingVisualization dpbv(singlePoleBalancing, !partiallyObservable, useAlphaBetaFilters, useDoubleExponentialSmoothing);
  dpbv.show();
  dpbv.resize(800, 400);

  return app.exec();
}