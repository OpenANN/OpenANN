#include <OpenANN/layers/LocalResponseNormalization.h>
#include <OpenANN/util/Random.h>

namespace OpenANN
{

LocalResponseNormalization::LocalResponseNormalization(
  OutputInfo info, double k, int n, double alpha, double beta)
  : I(info.outputs()), fm(info.dimensions[0]), rows(info.dimensions[1]),
    cols(info.dimensions[2]), fmSize(-1), x(0), denoms(1, I), y(1, I), etmp(1, I),
    e(1, I), k(k), n(n), alpha(alpha), beta(beta)
{
}

OutputInfo LocalResponseNormalization::initialize(
  std::vector<double*>& parameterPointers,
  std::vector<double*>& parameterDerivativePointers)
{
  fmSize = rows * cols;
  OPENANN_CHECK(fmSize > 0);

  OutputInfo info;
  info.dimensions.push_back(fm);
  info.dimensions.push_back(rows);
  info.dimensions.push_back(cols);
  return info;
}

void LocalResponseNormalization::forwardPropagate(Eigen::MatrixXd* x, Eigen::MatrixXd*& y, bool dropout)
{
  const int N = x->rows();
  this->y.conservativeResize(N, Eigen::NoChange);
  denoms.conservativeResize(N, Eigen::NoChange);
  this->x = x;

  #pragma omp parallel for
  for(int n = 0; n < N; n++)
  {
    for(int fmOut = 0, outputIdx = 0; fmOut < fm; fmOut++)
    {
      for(int r = 0; r < rows; r++)
      {
        for(int c = 0; c < cols; c++, outputIdx++)
        {
          double denom = 0.0;
          const int fmInMin = std::max(0, fmOut - n / 2);
          const int fmInMax = std::min(fm - 1, fmOut + n / 2);
          for(int fmIn = fmInMin; fmIn < fmInMax; fmIn++)
          {
            register double a = (*x)(n, fmIn * fmSize + r * cols + c);
            denom += a * a;
          }
          denom = k + alpha * denom;
          denoms(n, outputIdx) = denom;
          this->y(n, outputIdx) = (*x)(n, outputIdx) * std::pow(denom, -beta);
        }
      }
    }
  }
  y = &this->y;
}

void LocalResponseNormalization::backpropagate(Eigen::MatrixXd* ein,
                                               Eigen::MatrixXd*& eout,
                                               bool backpropToPrevious)
{
  const int N = y.rows();
  e.conservativeResize(N, Eigen::NoChange);
  etmp = (*ein).cwiseProduct(y).cwiseProduct(denoms.cwiseInverse()).array() *
         (-2.0 * alpha * beta);

  #pragma omp parallel for
  for(int n = 0; n < N; n++)
  {
    for(int fmOut = 0, outputIdx = 0; fmOut < fm; fmOut++)
    {
      for(int r = 0; r < rows; r++)
      {
        for(int c = 0; c < cols; c++, outputIdx++)
        {
          double nom = 0.0;
          const int fmInMin = std::max(0, fmOut - n / 2);
          const int fmInMax = std::min(fm - 1, fmOut + n / 2);
          for(int fmIn = fmInMin; fmIn < fmInMax; fmIn++)
            nom += etmp(fmIn * fmSize + r * cols + c);
          e(n, outputIdx) = (*x)(n, outputIdx) * nom + (*ein)(n, outputIdx) *
                            std::pow(denoms(n, outputIdx), -beta);
        }
      }
    }
  }

  eout = &e;
}

Eigen::MatrixXd& LocalResponseNormalization::getOutput()
{
  return y;
}

Eigen::VectorXd LocalResponseNormalization::getParameters()
{
  return Eigen::VectorXd();
}

}
