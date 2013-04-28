#pragma once
#if __GNUC__ >= 4
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <OpenANN/io/DataSet.h>
#include <OpenANN/io/Logger.h>
#include <Test/Stopwatch.h>

namespace OpenANN {

/**
 * @class DirectStorageDataSet
 *
 * Stores the inputs and outputs of the data set directly in two matrices.
 *
 * The data set can log results during optimization.
 */
class DirectStorageDataSet : public DataSet
{
public:
  enum LogInfo
  {
    NONE,
    MULTICLASS
  };

private:
  Eigen::MatrixXd* in;
  Eigen::MatrixXd* out;
  const int N;
  const int D;
  const int F;
  Eigen::VectorXd temporaryInput;
  Eigen::VectorXd temporaryOutput;
  LogInfo logInfo;
  Logger logger;
  int iteration;
  Stopwatch sw;

public:
  /**
   * Create an instance of DirectStorageDataSet.
   * @param in contains an instance in each column
   * @param out cointains a target in each column
   * @param logInfo activate evaluation of the model during optimization
   * @param target target of evaluation logger
   */
  DirectStorageDataSet(Eigen::MatrixXd* in, Eigen::MatrixXd* out = 0,
                       LogInfo logInfo = NONE, Logger::Target target = Logger::CONSOLE);
  virtual ~DirectStorageDataSet() {}
  virtual int samples() { return N; }
  virtual int inputs() { return D; }
  virtual int outputs() { return F; }
  virtual Eigen::VectorXd& getInstance(int i);
  virtual Eigen::VectorXd& getTarget(int i);
  virtual void finishIteration(Learner& learner);
};

}
