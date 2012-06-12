#include <optimization/SGD.h>
#include <Optimizable.h>
#include <StopCriteria.h>
#include <AssertionMacros.h>
#include <Random.h>
#include <Test/Stopwatch.h>
#include <numeric>
#include <vector>

namespace OpenANN {

SGD::SGD()
  : debugLogger(Logger::CONSOLE),
    regularize(false),
    regularizationCoefficient(0.0001)
{
}

SGD::~SGD()
{
}

void SGD::setOptimizable(Optimizable& opt)
{
  this->opt = &opt;
}

void SGD::setStopCriteria(const StopCriteria& stop)
{
  this->stop = stop;
}

void SGD::optimize()
{
  OPENANN_CHECK(opt->providesInitialization());

  RandomNumberGenerator rng;
  Vt gradient(opt->dimension());
  gradient.fill(0.0);
  Vt parameters = opt->currentParameters();
  const unsigned N = opt->examples();
  int iteration = 1;
  Stopwatch sw;

  int logAfterInstance = 60000;

  fpt initialLearningRate = 0.005;
  fpt learningRateDecay = 0.3;
  fpt minimalLearningRate = 0.00001;
  int learningRateDecayAfterEpoch = 20;
  fpt learningRate = initialLearningRate;

  const fpt initialGradientNorm = std::numeric_limits<fpt>::max() / (fpt) 10.0 / (fpt) N;
  fpt sumOfGradientNorms = (fpt) N * initialGradientNorm;
  std::vector<fpt> gradientNorms(N, initialGradientNorm);

  do {
    for(unsigned n = 0; n < N; n++)
    {
      unsigned index = 0;
      fpt random = rng.generate<fpt>(0.0, sumOfGradientNorms);
      fpt currentPointer = 0.0;
      for(unsigned i = 0; i < N; i++)
      {
        currentPointer += gradientNorms[i];
        if(currentPointer >= random)
        {
          index = i;
          break;
        }
      }

      gradient = opt->gradient(index);
      sumOfGradientNorms -= gradientNorms[index];
      gradientNorms[index] = gradient.norm();
      sumOfGradientNorms += gradientNorms[index];

      if(regularize)
        gradient -= regularizationCoefficient * parameters;

      parameters = opt->currentParameters() - learningRate * gradient;
      opt->setParameters(parameters);

      if(n % logAfterInstance == (unsigned) logAfterInstance-1 && n != N-1)
      {
        iteration++;
        debugLogger << ".";
      }
    }
    debugLogger << "\n";

    debugLogger << "iteration " << iteration++ << " finished in " << sw.stop(Stopwatch::MILLISECOND) << " ms\n";
    debugLogger << "learning rate " << learningRate << "\n";
    sumOfGradientNorms = std::accumulate(gradientNorms.begin(), gradientNorms.end(), 0.0);
    sw.start();
    opt->finishedIteration();
    if(debugLogger.isActive())
    {
      debugLogger << "test in " << sw.stop(Stopwatch::MILLISECOND) << " ms\n";
      debugLogger << "Gradient norm = " << sumOfGradientNorms << "\n";
      sw.start();
    }
    if(iteration % learningRateDecayAfterEpoch == 0)
    {
      learningRate *= learningRateDecay;
      if(learningRate < minimalLearningRate)
        learningRate = minimalLearningRate;
    }
  } while(stop.maximalIterations == StopCriteria::defaultValue.maximalFunctionEvaluations || iteration <= stop.maximalIterations);
}

Vt SGD::result()
{
  return optimum;
}

std::string SGD::name()
{
  return "Stochastic Gradient Descent";
}

}
