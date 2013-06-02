#ifndef DISTORTER_H_
#define DISTORTER_H_

#include <OpenANN/util/Random.h>
#include <OpenANN/util/AssertionMacros.h>
#include <OpenANN/util/EigenWrapper.h>
#include <Eigen/Dense>
#include <cmath>

/**
 * @class Distorter
 *
 * Creates distorted images.
 *
 * Available distortions are:
 *
 * - elastic distortions (emulation of uncontrolled oscillations of the hand
 *   muscles)
 * - rotation
 * - horizontal and vertical scaling
 *
 * To apply elastic distortions, the original image is distorted with random
 * displacements and then convolved with a Gaussian kernel of width
 * \f$ \sigma \in [ 5, 6] \f$. Then the values are rescaled by the factor
 * \f$ \alpha \in [36/255, 36/255] \f$. Afterwards the image will be rotated
 * by \f$ \beta \in [7.5, 15] \f$ degrees and scaled horizontally by the
 * factor \f$ \gamma_x \in [15, 20] \f$ and vertically by
 * \f$ \gamma_y \in [15, 20] \f$.
 *
 * Source: http://www.codeproject.com/Articles/16650/Neural-Network-for-Recognition-of-Handwritten-Digi
 */
class Distorter
{
public:
  //! elastic distortion
  double sigma;
  //! elastic scaling
  double alpha;
  //! maximal absolute rotation
  double beta;
  //! maximal horizontal scaling (percent)
  double gammaX;
  //! maximal vertical scaling (percent)
  double gammaY;

  //! has to be odd
  int gaussianKernelSize;
  Eigen::MatrixXd gaussianKernel;
  Eigen::MatrixXd distortionH, distortionV;

  Distorter(double sigma = 5.0, double alpha = 36.0/255.0, double beta = 15.0,
            double gammaX = 15.0, double gammaY = 15.0)
    : sigma(sigma), alpha(alpha), beta(beta), gammaX(gammaX), gammaY(gammaY),
      gaussianKernelSize(21), gaussianKernel(gaussianKernelSize,
                                             gaussianKernelSize)
  {
    const double twoSigmaSquared = 2.0 / (sigma * sigma);
    const double twoPiSigma = std::sqrt(2.0 * M_PI) / (sigma+1e-10);
    int center = gaussianKernelSize / 2;
    for(int row = 0; row < gaussianKernelSize; row++)
      for(int col = 0; col < gaussianKernelSize; col++)
        gaussianKernel(row, col) = twoPiSigma * std::exp(-twoSigmaSquared *
            (std::pow((double)(row - center), 2.0) +
             std::pow((double)(col - center), 2.0)));
    OPENANN_CHECK_MATRIX_BROKEN(gaussianKernel);
  }

  void createDistortionMap(int rows, int cols)
  {
    // Uniform random matrices in [-1, 1]
    Eigen::MatrixXd uniformH = Eigen::MatrixXd::Random(rows, cols);
    Eigen::MatrixXd uniformV = Eigen::MatrixXd::Random(rows, cols);

    // Gaussian filter
    distortionH.resize(rows, cols), distortionV.resize(rows, cols);
    distortionH.setZero();
    distortionV.setZero();
    int kernelCenter = gaussianKernelSize / 2;
    for(int r = 0; r < rows; r++)
    {
      for(int c = 0; c < cols; c++)
      {
        double convolvedH = 0.0, convolvedV = 0.0;
        for(int kr = 0; kr < gaussianKernelSize; kr++)
        {
          for(int kc = 0; kc < gaussianKernelSize; kc++)
          {
            int inputRow = r - kernelCenter + kr;
            int inputCol = c - kernelCenter + kc;
            if(inputRow >= 0 && inputRow < rows && inputCol >= 0 && inputCol < cols)
            {
              convolvedH += uniformH(inputRow, inputCol) * gaussianKernel(kr, kc);
              convolvedV += uniformV(inputRow, inputCol) * gaussianKernel(kr, kc);
            }
          }
        }
        distortionH(r, c) = alpha * convolvedH;
        distortionV(r, c) = alpha * convolvedV;
      }
    }
    OPENANN_CHECK_MATRIX_BROKEN(distortionH);
    OPENANN_CHECK_MATRIX_BROKEN(distortionV);

    // Image scaling
    OpenANN::RandomNumberGenerator rng;
    double horizontalScaling = rng.generate<double>(-1.0, 2.0) * gammaX / 100.0;
    double verticalScaling = rng.generate<double>(-1.0, 2.0) * gammaY / 100.0;
    OPENANN_CHECK_EQUALS(cols, rows); // could be generalized but YAGNI
    int imageCenter = rows / 2;
    for(int r = 0; r < rows; r++)
    {
      for(int c = 0; c < cols; c++)
      {
        distortionH(r, c) += horizontalScaling * (double)(c - imageCenter);
        distortionV(r, c) -= verticalScaling * (double)(imageCenter - r); // negative because of top-down bitmap
      }
    }
    OPENANN_CHECK_MATRIX_BROKEN(distortionH);
    OPENANN_CHECK_MATRIX_BROKEN(distortionV);

    // Rotation
    double angle = beta * rng.generate<double>(-1.0, 2.0) * M_PI / 180.0;
    double cosAngle = cos(angle);
    double sinAngle = sin(angle);

    for(int r = 0; r < rows; r++)
    {
      for(int c = 0; c < cols; c++)
      {
        distortionH(r, c) += (c - imageCenter) * (cosAngle - 1.0) - (imageCenter - r) * sinAngle;
        distortionV(r, c) -= (imageCenter - r) * (cosAngle - 1.0) - (c - imageCenter) * sinAngle;
      }
    }
    OPENANN_CHECK_MATRIX_BROKEN(distortionH);
    OPENANN_CHECK_MATRIX_BROKEN(distortionV);
  }


  void applyDistortions(Eigen::MatrixXd& instances, int rows, int cols)
  {
    Eigen::VectorXd instance;
    for(int n = 0; n < instances.cols(); n++)
    {
      instance = instances.col(n);
      applyDistortion(instance, rows, cols);
      instances.col(n) = instance;
    }
  }

  void applyDistortion(Eigen::VectorXd& instance, int rows, int cols)
  {
    createDistortionMap(rows, cols);
    Eigen::VectorXd input = instance;
    for(int r = 0; r < rows; r++)
    {
      for(int c = 0; c < cols; c++)
      {
        double sourceRow = (double) r - distortionV(r, c);
        double sourceCol = (double) c - distortionH(r, c);
        double rowFraction = sourceRow - ceil(sourceRow);
        double colFraction = sourceCol - ceil(sourceCol);
        double w1 = (1.0 - rowFraction) * (1.0 - colFraction);
        double w2 = (1.0 - rowFraction) * colFraction;
        double w3 = rowFraction * (1.0 - colFraction);
        double w4 = rowFraction * colFraction;

        if(!(sourceRow + 1 >= rows || sourceRow < 0 || sourceCol + 1 > cols || sourceCol < 0))
        {
          int sr = (int) sourceRow, sc = (int) sourceCol;
          int srn = sr + 1, scn = sc + 1;
          while(srn >= rows) srn -= rows;
          while(srn < 0) srn += rows;
          while(scn >= cols) scn -= cols;
          while(scn < 0) scn += cols;
          instance(r * cols + c) = w1 * input(sr * cols + sc)
                                 + w2 * input(sr * cols + scn)
                                 + w3 * input(srn * cols + sc)
                                 + w4 * input(srn * cols + scn);
        }
      }
    }
  }
};

#endif // DISTORTER_H_
