#include <io/FANNFormatLoader.h>
#include <io/DirectStorageDataSet.h>
#include <OpenANN>
#include <DeepNetwork.h>
#include <Test/Stopwatch.h>
#include <cstdlib>
#include <vector>
#ifdef PARALLEL_CORES
#include <omp.h>
#endif

/**
 * \page TwoSpiralsBenchmark Two Spirals
 *
 * This benchmark is based on the example program \ref TwoSpirals.
 *
 * The result will look like this:
 * \verbatim
Loaded training set.
Loaded test set.
Architecture: 2-20-10-1 (bias)
281 parameters
....................................................................................................
Finished 100 runs.
                Correct         Accuracy        Time/ms         Iterations
Mean+-StdDev    192.490+-0.795  0.997+-0.057    2575            313+-8.560
[min,max]       [190,193]       [0.984,1.000]                   [11,598]

Architecture: 2-20-20-1 (bias)
501 parameters
....................................................................................................
Finished 100 runs.
                Correct         Accuracy        Time/ms         Iterations
Mean+-StdDev    192.660+-0.705  0.998+-0.051    7176            251+-6.060
[min,max]       [190,193]       [0.984,1.000]                   [127,490]

Architecture: 2-20-20-1 (bias), Compression: 3-21-21
501 parameters
....................................................................................................
Finished 100 runs.
                Correct         Accuracy        Time/ms         Iterations
Mean+-StdDev    192.330+-0.843  0.997+-0.061    8344            286+-6.599
[min,max]       [189,193]       [0.979,1.000]                   [163,561]

Architecture: 2-20-20-1 (bias), Compression: 3-12-12
312 parameters
....................................................................................................
Finished 100 runs.
                Correct         Accuracy        Time/ms         Iterations
Mean+-StdDev    192.460+-0.805  0.997+-0.058    3430            323+-7.802
[min,max]       [190,193]       [0.984,1.000]                   [12,635]

Architecture: 2-20-20-1 (bias), Compression: 3-6-6
186 parameters
....................................................................................................
Finished 100 runs.
                Correct         Accuracy        Time/ms         Iterations
Mean+-StdDev    191.970+-0.941  0.995+-0.068    1740            401+-9.261
[min,max]       [184,193]       [0.953,1.000]                   [67,988]

Architecture: 2-20-20-1 (bias), Compression: 3-6-3
183 parameters
....................................................................................................
Finished 100 runs.
                Correct         Accuracy        Time/ms         Iterations
Mean+-StdDev    192.220+-0.883  0.996+-0.064    2042            484+-11.672
[min,max]       [186,193]       [0.964,1.000]                   [101,997]
   \endverbatim
 */

class EvaluatableDataset : public OpenANN::DirectStorageDataSet
{
public:
  int iterations;
  EvaluatableDataset(Mt& in, Mt& out)
    : DirectStorageDataSet(in, out), iterations(0)
  {}
  virtual void finishIteration(OpenANN::Learner& learner) { iterations++; }
};

struct Result
{
  int fp, tp, fn, tn, correct, wrong, iterations;
  fpt accuracy;

  Result()
    : fp(0), tp(0), fn(0), tn(0), correct(0), wrong(0), iterations(0),
      accuracy(0.0)
  {}
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
void setup(OpenANN::DeepNetwork& net, int architecture)
{
  OpenANN::Logger setupLogger(OpenANN::Logger::CONSOLE);
  setupLogger << "Architecture: ";
  switch(architecture)
  {
    case 0:
    {
      setupLogger << "2-20-10-1 (bias)\n";
      net.inputLayer(2)
        .fullyConnectedLayer(20, OpenANN::TANH)
        .fullyConnectedLayer(10, OpenANN::TANH)
        .outputLayer(1, OpenANN::TANH);
      break;
    }
    case 1:
    {
      setupLogger << "2-20-20-1 (bias)\n";
      net.inputLayer(2)
        .fullyConnectedLayer(20, OpenANN::TANH)
        .fullyConnectedLayer(20, OpenANN::TANH)
        .outputLayer(1, OpenANN::TANH);
      break;
    }
    case 2:
    {
      setupLogger << "2-20-20-1 (bias), Compression: 3-21-21\n";
      net.inputLayer(2)
        .compressedLayer(20, 3, OpenANN::TANH, "dct")
        .compressedLayer(20, 21, OpenANN::TANH, "dct")
        .compressedOutputLayer(1, 21, OpenANN::TANH, "dct");
      break;
    }
    case 3:
    {
      setupLogger << "2-20-20-1 (bias), Compression: 3-12-12\n";
      net.inputLayer(2)
        .compressedLayer(20, 3, OpenANN::TANH, "dct")
        .compressedLayer(20, 12, OpenANN::TANH, "dct")
        .compressedOutputLayer(1, 12, OpenANN::TANH, "dct");
      break;
    }
    case 4:
    {
      setupLogger << "2-20-20-1 (bias), Compression: 3-6-6\n";
      net.inputLayer(2)
        .compressedLayer(20, 3, OpenANN::TANH, "dct")
        .compressedLayer(20, 6, OpenANN::TANH, "dct")
        .compressedOutputLayer(1, 6, OpenANN::TANH, "dct");
      break;
    }
    case 5:
    {
      setupLogger << "2-20-20-1 (bias), Compression: 3-6-3\n";
      net.inputLayer(2)
        .compressedLayer(20, 3, OpenANN::TANH, "dct")
        .compressedLayer(20, 6, OpenANN::TANH, "dct")
        .compressedOutputLayer(1, 3, OpenANN::TANH, "dct");
      break;
    }
    default:
      setupLogger << "Unknown architecture, quitting.\n";
      exit(EXIT_FAILURE);
      break;
  }
  setupLogger << net.dimension() << " parameters\n";
}

/**
 * Evaluate the learned model.
 */
Result evaluate(OpenANN::DeepNetwork& net, OpenANN::FANNFormatLoader& loader,
                EvaluatableDataset& ds)
{
  Result result;
  for(int n = 0; n < loader.testN; n++)
  {
    fpt y = net(loader.testInput.col(n)).eval()(0);
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
  result.iterations = ds.iterations;
  return result;
}

/**
 * Print benchmark results.
 */
void logResults(std::vector<Result>& results, unsigned long time)
{
  typedef OpenANN::FloatingPointFormatter fmt;
  OpenANN::Logger resultLogger(OpenANN::Logger::CONSOLE);
  resultLogger << "\t\tCorrect\t\tAccuracy\tTime/ms\t\tIterations\n";
  Vt correct(results.size());
  Vt accuracy(results.size());
  Vt iterations(results.size());
  for(unsigned i = 0; i < results.size(); i++)
  {
    correct(i) = (fpt) results[i].correct;
    accuracy(i) = results[i].accuracy;
    iterations(i) = results[i].iterations;
  }
  fpt correctMean = correct.mean();
  fpt accuracyMean = accuracy.mean();
  fpt iterationsMean = iterations.mean();
  fpt correctMin = correct.minCoeff();
  fpt accuracyMin = accuracy.minCoeff();
  fpt iterationsMin = iterations.minCoeff();
  fpt correctMax = correct.maxCoeff();
  fpt accuracyMax = accuracy.maxCoeff();
  fpt iterationsMax = iterations.maxCoeff();
  for(unsigned i = 0; i < results.size(); i++)
  {
    correct(i) -= correctMean;
    accuracy(i) -= accuracyMean;
    iterations(i) -= iterationsMean;
  }
  correct = correct.cwiseAbs();
  accuracy = accuracy.cwiseAbs();
  iterations = iterations.cwiseAbs();
  fpt correctStdDev = std::sqrt(correct.mean());
  fpt accuracyStdDev = std::sqrt(accuracy.mean());
  fpt iterationsStdDev = std::sqrt(iterations.mean());
  resultLogger << "Mean+-StdDev\t";
  resultLogger << fmt(correctMean, 3) << "+-" << fmt(correctStdDev, 3) << "\t"
      << fmt(accuracyMean, 3) << "+-" << fmt(accuracyStdDev, 3) << "\t"
      << (int) ((fpt)time/(fpt)results.size()) << "\t\t"
      << iterationsMean << "+-" << fmt(iterationsStdDev, 3) << "\n";
  resultLogger << "[min,max]\t";
  resultLogger << "[" << correctMin << "," << correctMax << "]\t"
      << "[" << fmt(accuracyMin, 3) << "," << fmt(accuracyMax, 3) << "]\t\t\t"
      << "[" << (int) iterationsMin << "," << (int) iterationsMax << "]\n\n";
}

int main(int argc, char** argv)
{
#ifdef PARALLEL_CORES
  omp_set_num_threads(PARALLEL_CORES);
#endif

  OpenANN::Logger resultLogger(OpenANN::Logger::CONSOLE);
  const int architectures = 6;
  const int runs = 100;
  OpenANN::StopCriteria stop;
  stop.minimalSearchSpaceStep = 1e-5;
  stop.minimalValueDifferences = 1e-5;
  stop.maximalIterations = 1000;
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
    OpenANN::DeepNetwork net;
    setup(net, architecture);
    for(int run = 0; run < runs; run++)
    {
      EvaluatableDataset ds(loader.trainingInput, loader.trainingOutput);
      net.trainingSet(ds);
      sw.start();
      net.train(OpenANN::DeepNetwork::BATCH_LMA, OpenANN::DeepNetwork::SSE, stop);
      time += sw.stop(Stopwatch::MILLISECOND);
      Result result = evaluate(net, loader, ds);
      results.push_back(result);
      resultLogger << ".";
    }
    resultLogger << "\nFinished " << runs << " runs.\n";
    logResults(results, time);
  }
  return EXIT_SUCCESS;
}
