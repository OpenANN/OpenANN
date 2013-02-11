#include <io/FANNFormatLoader.h>
#include <io/DirectStorageDataSet.h>
#include <OpenANN>
#include <Test/Stopwatch.h>
#include <cstdlib>
#include <vector>
#ifdef PARALLEL_CORES
#include <omp.h>
#endif

/**
 * \page TwoSpiralsBenchmark Two Spirals
 *
 * This benchmark is based on the example program \ref TwoSpirals. It requires
 * the same data set and takes the directory of the data set as argument:
 * \verbatim ./TwoSpiralsBenchmark [directory] \endverbatim
 *
 * The result will look like this:
 * \verbatim
Loaded training set.
Loaded test set.
Architecture: 2-20-10-1 (bias)
281 parameters
....................................................................................................
Finished 100 runs.
		Correct		Accuracy	Time/ms		Iterations
Mean+-StdDev	188.250+-1.554	0.975+-0.112	5981		768+-21.979
[min,max]	[177,193]	[0.917,1.000]			[206,4342]

Architecture: 2-20-20-1 (bias)
501 parameters
....................................................................................................
Finished 100 runs.
		Correct		Accuracy	Time/ms		Iterations
Mean+-StdDev	188.570+-1.448	0.977+-0.104	12466		464+-13.867
[min,max]	[180,193]	[0.933,1.000]			[192,1216]

Architecture: 2-20-20-1 (bias), Compression: 3-21-21
501 parameters
....................................................................................................
Finished 100 runs.
		Correct		Accuracy	Time/ms		Iterations
Mean+-StdDev	186.060+-1.533	0.964+-0.110	8174		305+-9.903
[min,max]	[175,193]	[0.907,1.000]			[153,1038]

Architecture: 2-20-20-1 (bias), Compression: 3-12-12
312 parameters
....................................................................................................
Finished 100 runs.
		Correct		Accuracy	Time/ms		Iterations
Mean+-StdDev	185.660+-1.709	0.962+-0.123	5248		511+-16.075
[min,max]	[168,192]	[0.870,0.995]			[192,2886]

Architecture: 2-20-20-1 (bias), Compression: 3-6-6
186 parameters
....................................................................................................
Finished 100 runs.
		Correct		Accuracy	Time/ms		Iterations
Mean+-StdDev	184.750+-1.914	0.957+-0.138	3033		679+-18.572
[min,max]	[164,193]	[0.850,1.000]			[209,3023]

Architecture: 2-20-20-1 (bias), Compression: 3-6-3
183 parameters
....................................................................................................
Finished 100 runs.
		Correct		Accuracy	Time/ms		Iterations
Mean+-StdDev	185.140+-1.798	0.959+-0.129	3381		775+-20.821
[min,max]	[172,193]	[0.891,1.000]			[234,6584]
   \endverbatim
 */

class EvaluatableDataset : public OpenANN::DirectStorageDataSet
{
public:
  int iterations;
  EvaluatableDataset(Mt& in, Mt& out)
    : DirectStorageDataSet(in, out), iterations(0)
  {}
  virtual void finishIteration(OpenANN::MLP& mlp) { iterations++; }
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
    loader.trainingOutput(0, n) *= 2.0;
    loader.trainingOutput(0, n) -= 1.0;
  }
  for(int n = 0; n < loader.testN; n++)
  {
    loader.testOutput(0, n) *= 2.0;
    loader.testOutput(0, n) -= 1.0;
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
      setupLogger << "2-20-10-1 (bias)\n";
      mlp.input(2)
        .fullyConnectedHiddenLayer(20, OpenANN::MLP::TANH)
        .fullyConnectedHiddenLayer(10, OpenANN::MLP::TANH)
        .output(1, OpenANN::MLP::SSE, OpenANN::MLP::TANH);
      break;
    }
    case 1:
    {
      setupLogger << "2-20-20-1 (bias)\n";
      mlp.input(2)
        .fullyConnectedHiddenLayer(20, OpenANN::MLP::TANH)
        .fullyConnectedHiddenLayer(20, OpenANN::MLP::TANH)
        .output(1, OpenANN::MLP::SSE, OpenANN::MLP::TANH);
      break;
    }
    case 2:
    {
      setupLogger << "2-20-20-1 (bias), Compression: 3-21-21\n";
      mlp.input(2)
        .fullyConnectedHiddenLayer(20, OpenANN::MLP::TANH, 3)
        .fullyConnectedHiddenLayer(20, OpenANN::MLP::TANH, 21)
        .output(1, OpenANN::MLP::SSE, OpenANN::MLP::TANH, 21);
      break;
    }
    case 3:
    {
      setupLogger << "2-20-20-1 (bias), Compression: 3-12-12\n";
      mlp.input(2)
        .fullyConnectedHiddenLayer(20, OpenANN::MLP::TANH, 3)
        .fullyConnectedHiddenLayer(20, OpenANN::MLP::TANH, 12)
        .output(1, OpenANN::MLP::SSE, OpenANN::MLP::TANH, 12);
      break;
    }
    case 4:
    {
      setupLogger << "2-20-20-1 (bias), Compression: 3-6-6\n";
      mlp.input(2)
        .fullyConnectedHiddenLayer(20, OpenANN::MLP::TANH, 3)
        .fullyConnectedHiddenLayer(20, OpenANN::MLP::TANH, 6)
        .output(1, OpenANN::MLP::SSE, OpenANN::MLP::TANH, 6);
      break;
    }
    case 5:
    {
      setupLogger << "2-20-20-1 (bias), Compression: 3-6-3\n";
      mlp.input(2)
        .fullyConnectedHiddenLayer(20, OpenANN::MLP::TANH, 3)
        .fullyConnectedHiddenLayer(20, OpenANN::MLP::TANH, 6)
        .output(1, OpenANN::MLP::SSE, OpenANN::MLP::TANH, 3);
      break;
    }
    default:
      setupLogger << "Unknown architecture, quitting.\n";
      exit(EXIT_FAILURE);
      break;
  }
  setupLogger << mlp.dimension() << " parameters\n";
}

/**
 * Evaluate the learned model.
 */
Result evaluate(OpenANN::MLP& mlp, OpenANN::FANNFormatLoader& loader, EvaluatableDataset& ds)
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
  stop.minimalSearchSpaceStep = 1e-16;
  stop.minimalValueDifferences = 1e-16;
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
    OpenANN::MLP mlp(OpenANN::Logger::FILE, OpenANN::Logger::NONE);
    setup(mlp, architecture);
    for(int run = 0; run < runs; run++)
    {
      EvaluatableDataset ds(loader.trainingInput, loader.trainingOutput);
      mlp.trainingSet(ds);
      mlp.training(OpenANN::MLP::BATCH_LMA);
      sw.start();
      mlp.fit(stop);
      time += sw.stop(Stopwatch::MILLISECOND);
      Result result = evaluate(mlp, loader, ds);
      results.push_back(result);
      resultLogger << ".";
    }
    resultLogger << "\nFinished " << runs << " runs.\n";
    logResults(results, time);
  }
  return EXIT_SUCCESS;
}
