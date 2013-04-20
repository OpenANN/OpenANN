#pragma once

#include <OpenANN/Learner.h>
#include <Eigen/Dense>

namespace OpenANN {

class FiniteDifferences
{
public:
  Eigen::VectorXd inputGradient(const Eigen::VectorXd& x,
                                const Eigen::VectorXd& y, Learner& learner,
                                const double eps = 1e-2);
  Eigen::VectorXd parameterGradient(int n, Optimizable& opt,
                                    const double eps = 1e-2);
};

}
