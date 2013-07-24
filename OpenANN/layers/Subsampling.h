#ifndef OPENANN_LAYERS_SUBSAMPLING_H_
#define OPENANN_LAYERS_SUBSAMPLING_H_

#include <OpenANN/layers/Layer.h>
#include <OpenANN/ActivationFunctions.h>
#include <OpenANN/Regularization.h>

namespace OpenANN
{

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
 * Supports the following regularization types:
 *
 * - L1 penalty
 * - L2 penalty
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
  Eigen::MatrixXd* x;
  //! feature maps X output rows X output cols
  std::vector<Eigen::MatrixXd> W;
  std::vector<Eigen::MatrixXd> Wd;
  //! feature maps X output rows X output cols
  std::vector<Eigen::MatrixXd> Wb;
  std::vector<Eigen::MatrixXd> Wbd;
  Eigen::MatrixXd a;
  Eigen::MatrixXd y;
  Eigen::MatrixXd yd;
  Eigen::MatrixXd deltas;
  Eigen::MatrixXd e;
  int fmInSize, outRows, outCols, fmOutSize, maxRow, maxCol;
  Regularization regularization;

public:
  Subsampling(OutputInfo info, int kernelRows, int kernelCols, bool bias,
              ActivationFunction act, double stdDev,
              Regularization regularization);
  virtual OutputInfo initialize(std::vector<double*>& parameterPointers,
                                std::vector<double*>& parameterDerivativePointers);
  virtual void initializeParameters();
  virtual void updatedParameters() {}
  virtual void forwardPropagate(Eigen::MatrixXd* x, Eigen::MatrixXd*& y,
                                bool dropout);
  virtual void backpropagate(Eigen::MatrixXd* ein, Eigen::MatrixXd*& eout,
                             bool backpropToPrevious);
  virtual Eigen::MatrixXd& getOutput();
  virtual Eigen::VectorXd getParameters();
};

} // namespace OpenANN

#endif // OPENANN_LAYERS_SUBSAMPLING_H_
