#include <io/DirectStorageDataSet.h>
#include <AssertionMacros.h>

namespace OpenANN {

DirectStorageDataSet::DirectStorageDataSet(Mt& in, Mt& out)
  : in(&in), out(&out), N(in.cols()), D(in.rows()), F(out.rows()),
    temporaryInput(in.rows()), temporaryOutput(out.rows())
{
  OPENANN_CHECK(in.rows() > 0);
  OPENANN_CHECK(out.rows() > 0);
  OPENANN_CHECK_EQUALS(in.cols(), out.cols());
}

Vt& DirectStorageDataSet::getInstance(int i)
{
  OPENANN_CHECK_WITHIN(i, 0, in->cols()-1);
  temporaryInput = in->col(i);
  return temporaryInput;
}

Vt& DirectStorageDataSet::getTarget(int i)
{
  OPENANN_CHECK_WITHIN(i, 0, out->cols()-1);
  temporaryOutput = out->col(i);
  return temporaryOutput;
}

}
