#include <OpenANN/Evaluator.h>
#include <OpenANN/Evaluation.h>
#include <OpenANN/Learner.h>
#include <OpenANN/io/DataSet.h>
#include <OpenANN/util/Stopwatch.h>

namespace OpenANN
{

MulticlassEvaluator::MulticlassEvaluator(int interval, Logger::Target target)
  : interval(interval), logger(new Logger(target, "evaluation")),
    stopwatch(new Stopwatch), iteration(0)
{
  *logger << "# Multiclass problem\n"
          << "# it. SSE corr. wrong time (ms)\n\n";
  stopwatch->start();
}

MulticlassEvaluator::~MulticlassEvaluator()
{
  delete logger;
  delete stopwatch;
}

void MulticlassEvaluator::evaluate(Learner& learner, DataSet& dataSet)
{
  if(++iteration % interval == 0)
  {
    const int N = dataSet.samples();

    double e = 0.0;
    int correct = 0;
    int wrong = 0;
    Eigen::VectorXd temporaryOutput(dataSet.outputs());
    for(int n = 0; n < N; n++)
    {
      Eigen::VectorXd y = learner(dataSet.getInstance(n));
      temporaryOutput = dataSet.getTarget(n);
      e += (y - temporaryOutput).squaredNorm();
      int j1 = oneOfCDecoding(temporaryOutput);
      int j2 = oneOfCDecoding(y);
      if(j1 == j2)
        correct++;
      else
        wrong++;
    }
    e /= N;
    *logger << iteration << " " << e << " " << correct << " " << wrong << " "
        << stopwatch->stop(Stopwatch::MILLISECOND) << "\n";
  }
}

}
