#pragma once

#include <OpenANN/layers/Layer.h>
#include <OpenANN/ActivationFunctions.h>

namespace OpenANN {

/**
 * @class Subsampling
 *
 * Performs average pooling on 2D input feature maps.
 *
 * In a subsampling layer non-overlapping regions are combined to achieve
 * minor translation invariance and to reduce the number of nodes. Subsampling
 * was the only pooling layer in classical convolutional neural networks.
 *
 * The components of each region will be summed up, multiplied by a weight and
 * added to a bias to compute the activation of a neuron. Then we apply an
 * activation function.
 *
 * [1] Yann LeCun, Léon Bottou, Yoshua Bengio and Patrick Haffner:
 * Gradient-Based Learning Applied to Document Recognition,
 * Intelligent Signal Processing, IEEE Press, S. Haykin and B. Kosko (Eds.),
 * pp. 306-351, 2001.
 */
class Subsampling : public Layer
{
  int I, fm, inRows, inCols, kernelRows, kernelCols;
  bool bias;
  ActivationFunction act;
  double stdDev;
  Eigen::VectorXd* x;
  //! feature maps X output rows X output cols
  std::vector<Eigen::MatrixXd> W;
  std::vector<Eigen::MatrixXd> Wd;
  //! feature maps X output rows X output cols
  std::vector<Eigen::MatrixXd> Wb;
  std::vector<Eigen::MatrixXd> Wbd;
  Eigen::VectorXd a;
  Eigen::VectorXd y;
  Eigen::VectorXd yd;
  Eigen::VectorXd deltas;
  Eigen::VectorXd e;
  int fmInSize, outRows, outCols, fmOutSize, maxRow, maxCol;

public:
  Subsampling(OutputInfo info, int kernelRows, int kernelCols, bool bias,
              ActivationFunction act, double stdDev);
  virtual OutputInfo initialize(std::vector<double*>& parameterPointers,
                                std::vector<double*>& parameterDerivativePointers);
  virtual void initializeParameters();
  virtual void updatedParameters() {}
  virtual void forwardPropagate(Eigen::VectorXd* x, Eigen::VectorXd*& y,
                                bool dropout);
  virtual void backpropagate(Eigen::VectorXd* ein, Eigen::VectorXd*& eout);
  virtual Eigen::VectorXd& getOutput();
};

}
