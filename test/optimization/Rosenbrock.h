#include <optimization/Optimizable.h>
#include <AssertionMacros.h>

template<int N>
class Rosenbrock : public OpenANN::Optimizable
{
  Vt x;
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

  virtual Vt currentParameters()
  {
    return x;
  }

  virtual void setParameters(const Vt& parameters)
  {
    x = parameters;
  }

  fpt SQR(fpt t) { return t*t; }

  virtual fpt error()
  {
    fpt res(0);
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

  virtual Vt gradient()
  {
    OPENANN_CHECK(false && "Rosenbrock does not provide a gradient.");
    return Vt();
  }

  virtual bool providesHessian()
  {
    return false;
  }

  virtual Mt hessian()
  {
    OPENANN_CHECK(false && "Rosenbrock does not provide an hessian.");
    return Mt();
  }
};
