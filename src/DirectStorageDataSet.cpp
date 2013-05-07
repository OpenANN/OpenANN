#include <OpenANN/io/DirectStorageDataSet.h>
#include <OpenANN/optimization/Optimizable.h>
#include <OpenANN/util/AssertionMacros.h>
#include <OpenANN/Learner.h>

namespace OpenANN {

DirectStorageDataSet::DirectStorageDataSet(Eigen::MatrixXd* in,
                                           Eigen::MatrixXd* out,
                                           LogInfo logInfo,
                                           Logger::Target target)
  : in(in), out(out), N(in->rows()), D(in->cols()),
    F(out ? out->cols() : 0), temporaryInput(in->cols()),
    temporaryOutput(out ? out->cols() : 0), logInfo(logInfo),
    logger(target, "dataset"), iteration(0)
{
  OPENANN_CHECK(in->cols() > 0);
  OPENANN_CHECK(!out || in->rows() == out->rows());
  if(logInfo != NONE)
  {
    logger << "\n\n# Logging data set " << N << " x (" << D << " -> " << F << ").\n";
    if(logInfo == MULTICLASS)
      logger << "# Multiclass problem.\n\n";
    else
      logger << "# Unknown problem.\n\n";
  }
  sw.start();
}

Eigen::VectorXd& DirectStorageDataSet::getInstance(int i)
{
  OPENANN_CHECK_WITHIN(i, 0, in->rows()-1);
  temporaryInput = in->row(i);
  return temporaryInput;
}

Eigen::VectorXd& DirectStorageDataSet::getTarget(int i)
{
  OPENANN_CHECK(out != 0);
  OPENANN_CHECK_WITHIN(i, 0, out->rows()-1);
  temporaryOutput = out->row(i);
  return temporaryOutput;
}

void DirectStorageDataSet::finishIteration(Learner& learner)
{
  if(logInfo == MULTICLASS)
  {
    logger << ++iteration << " ";
    double e = 0.0;
    int correct = 0;
    int wrong = 0;
    for(int n = 0; n < N; n++)
    {
      temporaryInput = in->row(n);
      temporaryOutput = out->row(n);
      Eigen::VectorXd y = learner(temporaryInput);
      Eigen::VectorXd diff = y - temporaryOutput;
      e += diff.dot(diff);
      int j1 = 0;
      y.maxCoeff(&j1);
      int j2 = 0;
      temporaryOutput.maxCoeff(&j2);
      if(j1 == j2)
        correct++;
      else
        wrong++;
    }
    e /= N;
    logger << e << " " << correct << " " << wrong << " "
        << sw.stop(Stopwatch::MILLISECOND) << "\n";
  }
}

}
