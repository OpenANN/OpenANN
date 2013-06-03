#include <OpenANN/io/DirectStorageDataSet.h>
#include <OpenANN/optimization/Optimizable.h>
#include <OpenANN/util/AssertionMacros.h>
#include <OpenANN/Evaluator.h>

namespace OpenANN
{

DirectStorageDataSet::DirectStorageDataSet(Eigen::MatrixXd* in,
                                           Eigen::MatrixXd* out,
                                           Evaluator* evaluator)
  : in(in), out(out), N(in->rows()), D(in->cols()),
    F(out ? out->cols() : 0), temporaryInput(in->cols()),
    temporaryOutput(out ? out->cols() : 0), evaluator(evaluator)
{
  OPENANN_CHECK(in->cols() > 0);
  OPENANN_CHECK(!out || in->rows() == out->rows());
}

Eigen::VectorXd& DirectStorageDataSet::getInstance(int i)
{
  OPENANN_CHECK_WITHIN(i, 0, in->rows() - 1);
  temporaryInput = in->row(i);
  return temporaryInput;
}

Eigen::VectorXd& DirectStorageDataSet::getTarget(int i)
{
  OPENANN_CHECK(out != 0);
  OPENANN_CHECK_WITHIN(i, 0, out->rows() - 1);
  temporaryOutput = out->row(i);
  return temporaryOutput;
}

void DirectStorageDataSet::finishIteration(Learner& learner)
{
  if(evaluator)
    evaluator->evaluate(learner, *this);
}

}
