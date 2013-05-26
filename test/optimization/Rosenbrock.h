#include <OpenANN/optimization/Optimizable.h>
#include <OpenANN/util/AssertionMacros.h>

template<int N>
class Rosenbrock : public OpenANN::Optimizable
{
  Eigen::VectorXd x;
public:
  Rosenbrock()
      : x(N, 1)
  {
    for(int i = 0; i < N; i++)
      x(i) = 0;
  }

  virtual bool providesInitialization()
  {
    return false;
  }

  virtual void initialize()
  {
    OPENANN_CHECK(false && "Rosenbrock does not provide an initialization.");
  }

  virtual unsigned dimension()
  {
    return N;
  }

  virtual Eigen::VectorXd currentParameters()
  {
    return x;
  }

  virtual void setParameters(const Eigen::VectorXd& parameters)
  {
    x = parameters;
  }

  double SQR(double t) { return t*t; }

  virtual double error()
  {
    double res(0);
    for(int i = 0; i < N-1; i++)
    {
      res += SQR(1.0-x(i)) + 100.0*SQR(x(i+1)-SQR(x(i)));
    }
    return res;
  }

  virtual bool providesGradient()
  {
    return false;
  }

  virtual Eigen::VectorXd gradient()
  {
    OPENANN_CHECK(false && "Rosenbrock does not provide a gradient.");
    return Eigen::VectorXd();
  }
};
