#pragma once

#include <Eigen/Dense>
#include <io/Logger.h>

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
void createTwoSpiralsDataSet(int density, double maxDiameter, Mt& Xtr, Mt& Ytr, Mt& Xte, Mt& Yte)
{
  OpenANN::Logger logger(OpenANN::Logger::CONSOLE);
  // Number of interior data points per spiral to generate
  const int points = 96 * density;
  Xtr.resize(2, points);
  Ytr.resize(1, points);
  Xte.resize(2, points);
  Yte.resize(1, points);
  for(int i = 0; i <= points; i++)
  {
    // Angle is based on the iteration * PI/16, divided by point density
    const fpt angle = i * M_PI / (16.0 * density);
    // Radius is the maximum radius * the fraction of iterations left
    const fpt radius = maxDiameter * (104 * density - i) / (104 * density);
    // x and y are based upon cos and sin of the current radius
    const fpt x = radius * cos(angle);
    const fpt y = radius * sin(angle);

    logger << x << ", " << y << " -> " << 1 << "\n";
    logger << -x << ", " << -y << " -> " << -1 << "\n";
    if(i % 2 == 0)
    {
      Xtr(0, i) = x;
      Xtr(1, i) = y;
      Ytr(0, i) = 1.0;
      Xtr(0, i+1) = -x;
      Xtr(1, i+1) = -y;
      Ytr(0, i+1) = -1.0;
    }
    else
    {
      Xte(0, i-1) = x;
      Xte(1, i-1) = y;
      Yte(0, i-1) = 1.0;
      Xte(0, i) = -x;
      Xte(1, i) = -y;
      Yte(0, i) = -1.0;
    }
  }
  Mt shift = Mt::Ones(Xtr.rows(), Xtr.cols()) * 0.5;
  // Scaling
  Xtr /= 2.0;
  Xtr += shift;
  Xte /= 2.0;
  Xte += shift;
}