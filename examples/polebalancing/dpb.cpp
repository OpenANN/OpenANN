#include <iostream>
#include <QApplication>
#include <OpenANN>
#include "DoublePoleBalancingVisualization.h"

/**
 * \page PB Pole Balancing
 *
 * You can choose between the environments SinglePoleBalancing and
 * DoublePoleBalancing with or without velocities. The NeuroEvolutionAgent
 * solves this problem.
 *
 * \image html polebalancing.png
 */

void printUsage()
{
  std::cout << "unrecognized option, usage:" << std::endl
      << "\t-spb\tSingle Pole Balancing" << std::endl
      << "\t-po\tPartially Observable (without velocities)" << std::endl
      << "\t-ab\tUse alpha beta filters to estimate the velocities" << std::endl;
}

int main(int argc, char** argv)
{
  OpenANN::OpenANNLibraryInfo::print();

  bool singlePoleBalancing = false;
  bool partiallyObservable = false;
  bool useAlphaBetaFilters = false;
  for(int i = 1; i < argc; i++)
  {
    std::string argument(argv[i]);
    if(argument == std::string("-spb"))
      singlePoleBalancing = true;
    else if(argument == std::string("-po"))
      partiallyObservable = true;
    else if(argument == std::string("-ab"))
      useAlphaBetaFilters = true;
    else
      printUsage();
  }

  QApplication app(argc, argv);
  DoublePoleBalancingVisualization dpbv(singlePoleBalancing, !partiallyObservable, useAlphaBetaFilters);
  dpbv.show();
  dpbv.resize(800, 400);

  return app.exec();
}