#include <OpenANN>
#include "IDXLoader.h"
#ifdef PARALLEL_CORES
#include <omp.h>
#endif

int main(int argc, char** argv)
{
#ifdef PARALLEL_CORES
  omp_set_num_threads(PARALLEL_CORES);
#endif
  OpenANN::Logger interfaceLogger(OpenANN::Logger::CONSOLE);
  IDXLoader loader(28, 28, 60000, 10000);

  OpenANN::MLP mlp(OpenANN::Logger::APPEND_FILE, OpenANN::Logger::NONE);
  mlp.input(loader.D)
    .fullyConnectedHiddenLayer(200, OpenANN::MLP::TANH)
    .fullyConnectedHiddenLayer(100, OpenANN::MLP::TANH)
    .output(loader.F, OpenANN::MLP::CE, OpenANN::MLP::SM)
    .trainingSet(loader.trainingInput, loader.trainingOutput)
    .testSet(loader.testInput, loader.testOutput)
    .training(OpenANN::MLP::BATCH_SGD);
  OpenANN::StopCriteria stop;
  stop.maximalIterations = 30;
  interfaceLogger << "Created MLP.\n" << "D = " << loader.D << ", F = "
      << loader.F << ", N = " << loader.trainingN << ", L = " << mlp.dimension() << "\n";
  mlp.fit(stop);
  interfaceLogger << "Error = " << mlp.error() << "\n\n";
  interfaceLogger << "Wrote data to mlp-error.log.\n";

  return EXIT_SUCCESS;
}
