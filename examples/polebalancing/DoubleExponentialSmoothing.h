#ifndef DOUBLE_EXPONENTIAL_SMOOTHING_H_
#define DOUBLE_EXPONENTIAL_SMOOTHING_H_

#include <Eigen/Dense>
#include <OpenANN/io/Logger.h>

class DoubleExponentialSmoothing
{
  //! Data smoothing factor.
  double alpha;
  //! Trend smoothing factor.
  double beta;
  //! Current position.
  double xc;
  //! Next position.
  double xn;
  //! Current position estimation.
  double sc;
  //! Next position estimation.
  double sn;
  //! Current velocity estimation.
  double bc;
  //! Next velocity estimation.
  double bn;
  //! Current time step.
  int t;

public:
  DoubleExponentialSmoothing()
    : alpha(0.9), beta(0.9)
  {
    restart();
  }

  void restart()
  {
    xc = 0.0;
    xn = 0.0;
    t = 0;
  }

  Eigen::VectorXd operator()(double in)
  {
    xc = xn;
    xn = in;
    sc = sn;
    bc = bn;
    if(t == 0)
    {
      sn = in;
      bn = 0.0;
    }
    else if(t == 1)
    {
      sn = xc;
      bn = xn - xc;
    }
    else
    {
      sn = alpha * xn + (1.0 - alpha) * (sc + bc);
      bn = beta * (sn - sc) + (1.0 - beta) * bc;
    }
    t++;
    Eigen::VectorXd out(2);
    out << sn + bn, bn; // Forcast
    return out;
  }
};

#endif // DOUBLE_EXPONENTIAL_SMOOTHING_H_
