#pragma once

#include <Eigen/Dense>
#include <OpenANN/util/AssertionMacros.h>

/**
 * Creates two interlocked spirals that form different classes.
 *
 * Source is available at
 * <a href="http://www.cs.cmu.edu/afs/cs/project/ai-repository/ai/areas/neural/bench/0.html">
 * CMU Neural Networks Benchmark Collection</a>
 *
 * @param density higher densities result in greater data sets
 * @param maxDiameter diameter of the points on the outer side of the spirals
 * @param Xtr training inputs
 * @param Ytr training outputs
 * @param Xte test inputs
 * @param Yte test outputs
 */
void createTwoSpiralsDataSet(int density, double maxDiameter, Eigen::MatrixXd& Xtr, Eigen::MatrixXd& Ytr, Eigen::MatrixXd& Xte, Eigen::MatrixXd& Yte)
{
  // Number of interior data points per spiral to generate
  const int points = 96 * density;

  Xtr.resize(2, points+1);
  Ytr.resize(1, points+1);
  Xte.resize(2, points+1);
  Yte.resize(1, points+1);
  int trIdx = 0;
  int teIdx = 0;

  for(int i = 0; i <= points; i++)
  {
    // Angle is based on the iteration * PI/16, divided by point density
    const double angle = i * M_PI / (16.0 * density);
    // Radius is the maximum radius * the fraction of iterations left
    const double radius = maxDiameter * (104 * density - i) / (104 * density);
    // x and y are based upon cos and sin of the current radius
    const double x = radius * cos(angle);
    const double y = radius * sin(angle);

    if(i == points)
    {
      Xtr(0, trIdx) = x;
      Xtr(1, trIdx) = y;
      Ytr(0, trIdx) = 1.0;
      Xte(0, trIdx) = -x;
      Xte(1, teIdx) = -y;
      Yte(0, teIdx) = -1.0;
    }
    else if(i % 2 == 0)
    {
      OPENANN_CHECK_WITHIN(trIdx, 0, points);
      Xtr(0, trIdx) = x;
      Xtr(1, trIdx) = y;
      Ytr(0, trIdx) = 1.0;
      Xtr(0, trIdx+1) = -x;
      Xtr(1, trIdx+1) = -y;
      Ytr(0, trIdx+1) = -1.0;
      trIdx += 2;
    }
    else
    {
      OPENANN_CHECK_WITHIN(teIdx, 0, points);
      Xte(0, teIdx) = x;
      Xte(1, teIdx) = y;
      Yte(0, teIdx) = 1.0;
      Xte(0, teIdx+1) = -x;
      Xte(1, teIdx+1) = -y;
      Yte(0, teIdx+1) = -1.0;
      teIdx += 2;
    }
  }

  Eigen::MatrixXd shift = Eigen::MatrixXd::Ones(Xtr.rows(), Xtr.cols()) * 0.5;
  // Scaling
  Xtr /= 2.0;
  Xtr += shift;
  Xte /= 2.0;
  Xte += shift;
}