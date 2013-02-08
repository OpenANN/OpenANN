#include <CreateTwoSpiralsDataSet.h>
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
 * This benchmark is based on the example program \ref TwoSpirals. It requires
 * the same data set and takes the directory of the data set as argument:
 * \verbatim ./TwoSpiralsBenchmark [directory] \endverbatim
 *
 * The result will look like this:
 * \verbatim
$ ./TwoSpiralsBenchmark 
Architecture: 2-20-10-1 (bias)
281 parameters
....................................................................................................
Finished 100 runs.
		Correct		Accuracy	Time/ms		Iterations
Mean+-StdDev	188.650+-1.469	0.977+-0.106	6216		840+-21.797
[min,max]	[180,193]	[0.933,1.000]			[208,5156]

Architecture: 2-20-20-1 (bias)
501 parameters
....................................................................................................
Finished 100 runs.
		Correct		Accuracy	Time/ms		Iterations
Mean+-StdDev	188.450+-1.407	0.976+-0.101	11487		436+-13.035
[min,max]	[182,193]	[0.943,1.000]			[200,1247]

Architecture: 2-20-20-1 (bias), Compression: 3-21-21
501 parameters
....................................................................................................
Finished 100 runs.
		Correct		Accuracy	Time/ms		Iterations
Mean+-StdDev	185.610+-1.665	0.962+-0.120	7951		297+-9.877
[min,max]	[176,192]	[0.912,0.995]			[174,1886]

Architecture: 2-20-20-1 (bias), Compression: 3-12-12
312 parameters
....................................................................................................
Finished 100 runs.
		Correct		Accuracy	Time/ms		Iterations
Mean+-StdDev	186.090+-1.580	0.964+-0.114	4880		481+-14.788
[min,max]	[173,193]	[0.896,1.000]			[172,1608]

Architecture: 2-20-20-1 (bias), Compression: 3-6-6
186 parameters
....................................................................................................
Finished 100 runs.
		Correct		Accuracy	Time/ms		Iterations
Mean+-StdDev	184.680+-1.863	0.957+-0.134	3173		762+-19.506
[min,max]	[172,192]	[0.891,0.995]			[236,2175]

Architecture: 2-20-20-1 (bias), Compression: 3-6-3
183 parameters
....................................................................................................
Finished 100 runs.
		Correct		Accuracy	Time/ms		Iterations
Mean+-StdDev	184.690+-1.904	0.957+-0.137	3327		809+-20.544
[min,max]	[174,193]	[0.902,1.000]			[232,3246]
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
Result evaluate(OpenANN::DeepNetwork& net, const Mt& testInput,
		const Mt& testOutput, EvaluatableDataset& ds)
{
  Result result;
  for(int n = 0; n < testInput.cols(); n++)
  {
    fpt y = net(testInput.col(n)).eval()(0);
    fpt t = testOutput(0, n);
    if(y > 0.0 && t > 0.0)
      result.tp++;
    else if(y > 0.0 && t < 0.0)
      result.fp++;
    else if(y < 0.0 && t > 0.0)
      result.fn++;
    else
      result.tn++;
  }
  result.correct = result.tn + result.tp;
  result.wrong = result.fn + result.fp;
  result.accuracy = (fpt) result.correct / (fpt) testInput.cols();
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

  Mt Xtr, Ytr, Xte, Yte;
  createTwoSpiralsDataSet(2, 1.0, Xtr, Ytr, Xte, Yte);

  for(int architecture = 0; architecture < architectures; architecture++)
  {
    long unsigned time = 0;
    std::vector<Result> results;
    OpenANN::DeepNetwork net(OpenANN::DeepNetwork::SSE);
    setup(net, architecture);
    for(int run = 0; run < runs; run++)
    {
      EvaluatableDataset ds(Xtr, Ytr);
      net.trainingSet(ds);
      sw.start();
      net.train(OpenANN::DeepNetwork::BATCH_LMA, stop);
      time += sw.stop(Stopwatch::MILLISECOND);
      Result result = evaluate(net, Xte, Yte, ds);
      results.push_back(result);
      resultLogger << ".";
    }
    resultLogger << "\nFinished " << runs << " runs.\n";
    logResults(results, time);
  }
  return EXIT_SUCCESS;
}
