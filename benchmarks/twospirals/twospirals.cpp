#include <io/FANNFormatLoader.h>
#include <OpenANN>
#include <Test/Stopwatch.h>
#include <cstdlib>
#ifdef PARALLEL_CORES
#include <omp.h>
#endif

struct Result
{
  int fp, tp, fn, tn, correct, wrong;
  fpt accuracy;
};

/**
 * Scale the desired output to [-1,1].
 */
void preprocess(OpenANN::FANNFormatLoader& loader)
{
  for(int n = 0; n < loader.trainingN; n++)
  {
    loader.trainingOutput(0, n) * 2.0;
    loader.trainingOutput(0, n) - 1.0;
  }
  for(int n = 0; n < loader.testN; n++)
  {
    loader.testOutput(0, n) * 2.0;
    loader.testOutput(0, n) - 1.0;
  }
}

/**
 * Set up the desired MLP architecture.
 */
void setup(OpenANN::MLP& mlp, int architecture)
{
  OpenANN::Logger setupLogger(OpenANN::Logger::CONSOLE);
  switch(architecture)
  {
    case 0:
    {
      mlp.input(2)
        .fullyConnectedHiddenLayer(10, OpenANN::MLP::TANH)
        .fullyConnectedHiddenLayer(5, OpenANN::MLP::TANH)
        .output(1, OpenANN::MLP::SSE, OpenANN::MLP::TANH);
      break;
    }
    default:
      setupLogger << "Unknown architecture, quitting.\n";
      exit(EXIT_FAILURE);
  }
}

/**
 * Evaluate the learned model.
 */
Result evaluate(OpenANN::MLP& mlp, OpenANN::FANNFormatLoader& loader)
{
  Result result = {0, 0, 0, 0, 0, 0, 0.0};
  for(int n = 0; n < loader.testN; n++)
  {
    fpt y = mlp(loader.testInput.col(n)).eval()(0);
    fpt t = loader.testOutput(0, n);
    if(y > 0 && t > 0)
      result.tp++;
    else if(y > 0 && t < 0)
      result.fp++;
    else if(y < 0 && t > 0)
      result.fn++;
    else
      result.tn++;
  }
  result.correct = result.tn + result.tp;
  result.wrong = result.fn + result.fp;
  result.accuracy = (fpt) result.correct / (fpt) loader.testN;
  return result;
}

int main(int argc, char** argv)
{
#ifdef PARALLEL_CORES
  omp_set_num_threads(PARALLEL_CORES);
#endif

  OpenANN::Logger resultLogger(OpenANN::Logger::CONSOLE);
  const int architectures = 8;
  const int runs = 100;
  OpenANN::StopCriteria stop;
  stop.minimalSearchSpaceStep = 1e-5;
  stop.maximalIterations = 10000;
  Stopwatch sw;

  std::string directory = ".";
  if(argc > 1)
    directory = std::string(argv[1]);
  OpenANN::FANNFormatLoader loader(directory + "/two-spiral");
  preprocess(loader);

  for(int architecture = 0; architecture < architectures; architecture++)
  {
    long unsigned time = 0;
    OpenANN::MLP mlp(OpenANN::Logger::NONE, OpenANN::Logger::NONE);
    setup(mlp, architecture);
    mlp.trainingSet(loader.trainingInput, loader.trainingOutput);
    mlp.training(OpenANN::MLP::BATCH_LMA);
    for(int run = 0; run < runs; run++)
    {
      sw.start();
      mlp.fit(stop);
      time += sw.stop(Stopwatch::MILLISECOND);
      Result result = evaluate(mlp, loader);
      resultLogger << "Run " << run << ": " << time << " ms, " << (result.accuracy * 100) << "%.\n";
    }
  }
  return EXIT_SUCCESS;
}
