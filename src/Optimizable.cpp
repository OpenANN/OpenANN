#include <OpenANN/optimization/Optimizable.h>
#include <OpenANN/util/AssertionMacros.h>

namespace OpenANN {

void Optimizable::VJ(Eigen::VectorXd& values, Eigen::MatrixXd& jacobian)
{
  OPENANN_CHECK_EQUALS(values.rows(), (int) examples());
  OPENANN_CHECK_EQUALS(jacobian.rows(), (int) examples());
  OPENANN_CHECK_EQUALS(jacobian.cols(), (int) dimension());
  for(unsigned n = 0; n < examples(); n++)
  {
    values(n) = error(n);
    jacobian.row(n) = gradient(n);
  }
}

}
