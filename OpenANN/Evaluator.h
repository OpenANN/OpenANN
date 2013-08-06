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
  virtual void evaluate(Learner& learner, DataSet& dataSet) = 0;
};

class MulticlassEvaluator : public Evaluator
{
  int interval;
  Logger* logger;
  Stopwatch* stopwatch;
  int iteration;
public:
  MulticlassEvaluator(int interval = 1,
                      Logger::Target target = Logger::CONSOLE);
  virtual ~MulticlassEvaluator();
  virtual void evaluate(Learner& learner, DataSet& dataSet);
};

} // namespace OpenANN

#endif // OPENANN_EVALUATOR_H_
