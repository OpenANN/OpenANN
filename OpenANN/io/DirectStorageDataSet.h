#pragma once
#if __GNUC__ >= 4
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <io/DataSet.h>
#include "Logger.h"
#include <Test/Stopwatch.h>

namespace OpenANN {

class DirectStorageDataSet : public DataSet
{
public:
  enum LogInfo
  {
    NONE,
    MULTICLASS
  };

private:
  Mt* in;
  Mt* out;
  const int N;
  const int D;
  const int F;
  Vt temporaryInput;
  Vt temporaryOutput;
  LogInfo logInfo;
  Logger logger;
  int iteration;
  Stopwatch sw;

public:
  DirectStorageDataSet(Mt& in, Mt& out, LogInfo logInfo = NONE,
                       Logger::Target target = Logger::CONSOLE);
  virtual ~DirectStorageDataSet() {}
  virtual int samples() { return N; }
  virtual int inputs() { return D; }
  virtual int outputs() { return F; }
  virtual Vt& getInstance(int i);
  virtual Vt& getTarget(int i);
  virtual void finishIteration(Learner& learner);
};

}
