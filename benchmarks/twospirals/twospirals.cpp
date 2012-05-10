#include <io/FANNFormatLoader.h>
#include <OpenANN>
#include <Test/Stopwatch.h>
#include <cstdlib>
#include <vector>
#ifdef PARALLEL_CORES
#include <omp.h>
#endif

struct Result
{
  int fp, tp, fn, tn, correct, wrong;
  fpt accuracy;
  int accumulated;

  Result()
    : fp(0), tp(0), fn(0), tn(0), correct(0), wrong(0), accuracy(0.0),
      accumulated(1)
  {
  }

  void operator+=(Result& other)
  {
    fp += other.fp;
    tp += other.tp;
    fn += other.fn;
    tn += other.tn;
    correct += other.correct;
    wrong += other.wrong;
    accuracy += other.accuracy;
    accumulated += other.accumulated;
  }
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
  setupLogger << "Architecture: ";
  switch(architecture)
  {
    case 0:
    {
      setupLogger << "2-10-5-1 (bias)\n";
      mlp.input(2)
        .fullyConnectedHiddenLayer(10, OpenANN::MLP::TANH)
        .fullyConnectedHiddenLayer(5, OpenANN::MLP::TANH)
        .output(1, OpenANN::MLP::SSE, OpenANN::MLP::TANH);
      break;
    }
    case 1:
    {
      setupLogger << "2-10-10-1 (bias)\n";
      mlp.input(2)
        .fullyConnectedHiddenLayer(10, OpenANN::MLP::TANH)
        .fullyConnectedHiddenLayer(10, OpenANN::MLP::TANH)
        .output(1, OpenANN::MLP::SSE, OpenANN::MLP::TANH);
      break;
    }
    case 2:
    {
      setupLogger << "2-20-10-1 (bias)\n";
      mlp.input(2)
        .fullyConnectedHiddenLayer(20, OpenANN::MLP::TANH)
        .fullyConnectedHiddenLayer(10, OpenANN::MLP::TANH)
        .output(1, OpenANN::MLP::SSE, OpenANN::MLP::TANH);
      break;
    }
    case 3:
    {
      setupLogger << "2-20-20-1 (bias)\n";
      mlp.input(2)
        .fullyConnectedHiddenLayer(20, OpenANN::MLP::TANH)
        .fullyConnectedHiddenLayer(20, OpenANN::MLP::TANH)
        .output(1, OpenANN::MLP::SSE, OpenANN::MLP::TANH);
      break;
    }
    case 4:
    {
      setupLogger << "2-20-20-1 (bias), Compression: 3-21-21\n";
      mlp.input(2)
        .fullyConnectedHiddenLayer(20, OpenANN::MLP::TANH, 3)
        .fullyConnectedHiddenLayer(20, OpenANN::MLP::TANH, 21)
        .output(1, OpenANN::MLP::SSE, OpenANN::MLP::TANH, 21);
      break;
    }
    case 5:
    {
      setupLogger << "2-20-20-1 (bias), Compression: 3-12-12\n";
      mlp.input(2)
        .fullyConnectedHiddenLayer(20, OpenANN::MLP::TANH, 3)
        .fullyConnectedHiddenLayer(20, OpenANN::MLP::TANH, 12)
        .output(1, OpenANN::MLP::SSE, OpenANN::MLP::TANH, 12);
      break;
    }
    case 6:
    {
      setupLogger << "2-20-20-1 (bias), Compression: 3-6-6\n";
      mlp.input(2)
        .fullyConnectedHiddenLayer(20, OpenANN::MLP::TANH, 3)
        .fullyConnectedHiddenLayer(20, OpenANN::MLP::TANH, 6)
        .output(1, OpenANN::MLP::SSE, OpenANN::MLP::TANH, 6);
      break;
    }
    case 7:
    {
      setupLogger << "2-20-20-1 (bias), Compression: 3-6-1\n";
      mlp.input(2)
        .fullyConnectedHiddenLayer(20, OpenANN::MLP::TANH, 3)
        .fullyConnectedHiddenLayer(20, OpenANN::MLP::TANH, 6)
        .output(1, OpenANN::MLP::SSE, OpenANN::MLP::TANH, 1);
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
  Result result;
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

void logResults(std::vector<Result>& results, unsigned long time)
{
  OpenANN::Logger resultLogger(OpenANN::Logger::CONSOLE);
  resultLogger << "\t\tCorrect\t\tAccuracy\t\tTime/ms\n";
  Vt correct(results.size());
  Vt accuracy(results.size());
  for(unsigned i = 0; i < results.size(); i++)
  {
    correct(i) = (fpt) results[i].correct;
    accuracy(i) = results[i].accuracy;
  }
  fpt correctMean = correct.mean();
  fpt accuracyMean = accuracy.mean();
  fpt correctMin = correct.minCoeff();
  fpt accuracyMin = accuracy.minCoeff();
  fpt correctMax = correct.maxCoeff();
  fpt accuracyMax = accuracy.maxCoeff();
  for(unsigned i = 0; i < results.size(); i++)
  {
    correct(i) -= correctMean;
    accuracy(i) -= accuracyMean;
  }
  correct = correct.cwiseAbs();
  accuracy = accuracy.cwiseAbs();
  fpt correctStdDev = std::sqrt(correct.mean());
  fpt accuracyStdDev = std::sqrt(accuracy.mean());
  resultLogger << "Mean+-StdDev\t";
  resultLogger << correctMean << "+-" << correctStdDev << "\t"
      << accuracyMean << "+-" << correctStdDev << "\t"
      << (fpt)time/(fpt)results.size() << "\n";
  resultLogger << "[min,max]\t";
  resultLogger << "[" << correctMin << "," << correctMax << "]\t"
      << "[" << accuracyMin << "," << accuracyMax << "]\n\n";
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
  stop.minimalSearchSpaceStep = 1e-10;
  stop.minimalValueDifferences = 1e-10;
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
    std::vector<Result> results;
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
      results.push_back(result);
      resultLogger << ".";
    }
    resultLogger << "\nFinished " << runs << " runs.\n";
    logResults(results, time);
  }
  return EXIT_SUCCESS;
}
