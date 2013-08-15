#ifndef OPENANN_EVALUATOR_H_
#define OPENANN_EVALUATOR_H_

#include <OpenANN/io/Logger.h>

class Stopwatch;

namespace OpenANN
{

class Learner;
class DataSet;

/**
 * @class Evaluator
 *
 * Evaluates a Learner.
 *
 * This is usually required to monitor the optimization progress of Learners.
 */
class Evaluator
{
public:
  virtual ~Evaluator() {}
  /**
   * Evaluate learner on data set.
   * @param learner current model
   * @param dataSet validation set
   */
  virtual void evaluate(Learner& learner, DataSet& dataSet) = 0;
};

/**
 * @class MulticlassEvaluator
 *
 * Evaluates learners for multiclass problems.
 *
 * The following metrics will be computed:
 *
 * - SSE
 * - correct predictions
 * - wrong predictions
 *
 * In addition, the number of iteration and the elapsed time will be logged. The
 * logger will be called "evaluation", i.e. the corresponding log file is
 * "evaluation.log" or "evaluation-date.log".
 */
class MulticlassEvaluator : public Evaluator
{
  int interval;
  Logger* logger;
  Stopwatch* stopwatch;
  int iteration;
public:
  /**
   * Create MulticlassEvaluator.
   * @param interval logging interval, the learner will be evaluated after
   *                 *interval* iterations
   * @param target target of the logger
   */
  MulticlassEvaluator(int interval = 1,
                      Logger::Target target = Logger::CONSOLE);
  virtual ~MulticlassEvaluator();
  virtual void evaluate(Learner& learner, DataSet& dataSet);
};

} // namespace OpenANN

#endif // OPENANN_EVALUATOR_H_
