#include <CreateTwoSpiralsDataSet.h>
#include <OpenANN/io/DirectStorageDataSet.h>
#include <OpenANN/OpenANN>
#include <Test/Stopwatch.h>
#include <cstdlib>
#include <vector>

/**
 * \page TwoSpiralsBenchmark Two Spirals
 *
 * This benchmark is based on the example program \ref TwoSpirals.
 *
 * The result will look like this:
 * \verbatim
$ ./TwoSpiralsBenchmark
Architecture: 2-20-10-1 (bias)
281 parameters
....................................................................................................
Finished 100 runs.
    Correct   Accuracy  Time/ms   Iterations
Mean+-StdDev  188.250+-1.554  0.975+-0.112  5981    768+-21.979
[min,max] [177,193] [0.917,1.000]     [206,4342]

Architecture: 2-20-20-1 (bias)
501 parameters
....................................................................................................
Finished 100 runs.
    Correct   Accuracy  Time/ms   Iterations
Mean+-StdDev  188.570+-1.448  0.977+-0.104  12466   464+-13.867
[min,max] [180,193] [0.933,1.000]     [192,1216]

Architecture: 2-20-20-1 (bias), Compression: 3-21-21
501 parameters
....................................................................................................
Finished 100 runs.
    Correct   Accuracy  Time/ms   Iterations
Mean+-StdDev  186.060+-1.533  0.964+-0.110  8174    305+-9.903
[min,max] [175,193] [0.907,1.000]     [153,1038]

Architecture: 2-20-20-1 (bias), Compression: 3-12-12
312 parameters
....................................................................................................
Finished 100 runs.
    Correct   Accuracy  Time/ms   Iterations
Mean+-StdDev  185.660+-1.709  0.962+-0.123  5248    511+-16.075
[min,max] [168,192] [0.870,0.995]     [192,2886]

Architecture: 2-20-20-1 (bias), Compression: 3-6-6
186 parameters
....................................................................................................
Finished 100 runs.
    Correct   Accuracy  Time/ms   Iterations
Mean+-StdDev  184.750+-1.914  0.957+-0.138  3033    679+-18.572
[min,max] [164,193] [0.850,1.000]     [209,3023]

Architecture: 2-20-20-1 (bias), Compression: 3-6-3
183 parameters
....................................................................................................
Finished 100 runs.
    Correct   Accuracy  Time/ms   Iterations
Mean+-StdDev  185.140+-1.798  0.959+-0.129  3381    775+-20.821
[min,max] [172,193] [0.891,1.000]     [234,6584]
   \endverbatim
 */

class EvaluatableDataset : public OpenANN::DirectStorageDataSet
{
public:
  int iterations;
  EvaluatableDataset(Eigen::MatrixXd& in, Eigen::MatrixXd& out)
    : DirectStorageDataSet(&in, &out), iterations(0)
  {}
  virtual void finishIteration(OpenANN::Learner& learner) { iterations++; }
};

struct Result
{
  int fp, tp, fn, tn, correct, wrong, iterations;
  double accuracy;

  Result()
    : fp(0), tp(0), fn(0), tn(0), correct(0), wrong(0), iterations(0),
      accuracy(0.0)
  {}
};

/**
 * Set up the desired MLP architecture.
 */
void setup(OpenANN::Net& net, int architecture)
{
  OpenANN::Log::getLevel() = OpenANN::Log::INFO;
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
Result evaluate(OpenANN::Net& net, const Eigen::MatrixXd& testInput, const Eigen::MatrixXd& testOutput,
                EvaluatableDataset& ds)
{
  Result result;
  Eigen::VectorXd input(testInput.cols());
  for(int n = 0; n < testInput.rows(); n++)
  {
    input = testInput.row(n);
    double y = net(input)(0);
    double t = testOutput(n, 0);
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
  result.accuracy = (double) result.correct / (double) testInput.rows();
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
  Eigen::VectorXd correct(results.size());
  Eigen::VectorXd accuracy(results.size());
  Eigen::VectorXd iterations(results.size());
  for(unsigned i = 0; i < results.size(); i++)
  {
    correct(i) = (double) results[i].correct;
    accuracy(i) = results[i].accuracy;
    iterations(i) = results[i].iterations;
  }
  double correctMean = correct.mean();
  double accuracyMean = accuracy.mean();
  double iterationsMean = iterations.mean();
  double correctMin = correct.minCoeff();
  double accuracyMin = accuracy.minCoeff();
  double iterationsMin = iterations.minCoeff();
  double correctMax = correct.maxCoeff();
  double accuracyMax = accuracy.maxCoeff();
  double iterationsMax = iterations.maxCoeff();
  for(unsigned i = 0; i < results.size(); i++)
  {
    correct(i) -= correctMean;
    accuracy(i) -= accuracyMean;
    iterations(i) -= iterationsMean;
  }
  correct = correct.cwiseAbs();
  accuracy = accuracy.cwiseAbs();
  iterations = iterations.cwiseAbs();
  double correctStdDev = std::sqrt(correct.mean());
  double accuracyStdDev = std::sqrt(accuracy.mean());
  double iterationsStdDev = std::sqrt(iterations.mean());
  resultLogger << "Mean+-StdDev\t";
  resultLogger << fmt(correctMean, 3) << "+-" << fmt(correctStdDev, 3) << "\t"
               << fmt(accuracyMean, 3) << "+-" << fmt(accuracyStdDev, 3) << "\t"
               << (int)((double)time / (double)results.size()) << "\t\t"
               << iterationsMean << "+-" << fmt(iterationsStdDev, 3) << "\n";
  resultLogger << "[min,max]\t";
  resultLogger << "[" << correctMin << "," << correctMax << "]\t"
               << "[" << fmt(accuracyMin, 3) << "," << fmt(accuracyMax, 3) << "]\t\t\t"
               << "[" << (int) iterationsMin << "," << (int) iterationsMax << "]\n\n";
}

int main(int argc, char** argv)
{
  OpenANN::useAllCores();

  OpenANN::Logger resultLogger(OpenANN::Logger::CONSOLE);
  const int architectures = 6;
  const int runs = 100;
  OpenANN::StoppingCriteria stop;
  stop.minimalSearchSpaceStep = 1e-16;
  stop.minimalValueDifferences = 1e-16;
  stop.maximalIterations = 10000;
  Stopwatch sw;

  Eigen::MatrixXd Xtr, Ytr, Xte, Yte;
  createTwoSpiralsDataSet(2, 1.0, Xtr, Ytr, Xte, Yte);

  for(int architecture = 0; architecture < architectures; architecture++)
  {
    long unsigned time = 0;
    std::vector<Result> results;
    OpenANN::Net net;
    setup(net, architecture);
    for(int run = 0; run < runs; run++)
    {
      EvaluatableDataset ds(Xtr, Ytr);
      net.trainingSet(ds);
      sw.start();
      OpenANN::train(net, "LMA", OpenANN::MSE, stop, true);
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
