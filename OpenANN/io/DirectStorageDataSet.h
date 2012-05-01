#pragma once
#if __GNUC__ >= 4
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <io/DataSet.h>

namespace OpenANN {

class DirectStorageDataSet : public DataSet
{
  Mt* in;
  Mt* out;
  const int N;
  const int D;
  const int F;
  Vt temporaryInput;
  Vt temporaryOutput;
public:
  DirectStorageDataSet(Mt& in, Mt& out);
  virtual ~DirectStorageDataSet() {}
  virtual int samples() { return N; }
  virtual int inputs() { return D; }
  virtual int outputs() { return F; }
  virtual Vt& getInstance(int i);
  virtual Vt& getTarget(int i);
  virtual void finishIteration(MLP& mlp) {}
};

}
