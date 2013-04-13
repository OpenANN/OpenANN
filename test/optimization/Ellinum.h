#include <OpenANN/optimization/Optimizable.h>
#include <OpenANN/util/AssertionMacros.h>
#include <cmath>

template<int N>
class Ellinum : public OpenANN::Optimizable
{
  Vt x;
public:
  Ellinum()
      : x(N, 1)
  {
  }

  virtual bool providesInitialization()
  {
    return false;
  }

  virtual void initialize()
  {
    OPENANN_CHECK(false && "Ellinum does not provide an initialization.");
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

  virtual fpt error()
  {
    fpt sum(0);
    static fpt maxVerhaeltnis = 0.0;
    if(maxVerhaeltnis == 0.0)
    {
      for (maxVerhaeltnis = 1.0; 
        maxVerhaeltnis < 1e99 && maxVerhaeltnis < 2. * maxVerhaeltnis; 
        maxVerhaeltnis *= 2.)
      if (maxVerhaeltnis == maxVerhaeltnis + 1.)
        break;
      maxVerhaeltnis *= 10.;
      maxVerhaeltnis = sqrt (maxVerhaeltnis);
    }
    if (N < 3)
      return x(0,0) * x(0,0);
    for(int i = 1; i < N; ++i)
      sum += exp(log(maxVerhaeltnis) * 2. * (fpt)(i-1)/(N-2)) * x(i, 0)*x(i, 0);
    return sum;
  }

  virtual bool providesGradient()
  {
    return false;
  }

  virtual Vt gradient()
  {
    OPENANN_CHECK(false && "Ellinum does not provide a gradient.");
    return Vt();
  }

  virtual bool providesHessian()
  {
    return false;
  }

  virtual Mt hessian()
  {
    OPENANN_CHECK(false && "Ellinum does not provide an hessian.");
    return Mt();
  }
};
