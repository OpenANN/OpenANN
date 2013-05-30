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
  Logger* logger;
  Stopwatch* stopwatch;
  int iteration;
public:
  MulticlassEvaluator(Logger::Target target = Logger::CONSOLE);
  virtual ~MulticlassEvaluator();
  virtual void evaluate(Learner& learner, DataSet& dataSet);
};

} // OpenANN

#endif // OPENANN_EVALUATOR_H_
