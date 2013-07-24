#ifndef OPENANN_LAYERS_CONVOLUTIONAL_H_
#define OPENANN_LAYERS_CONVOLUTIONAL_H_

#include <OpenANN/layers/Layer.h>
#include <OpenANN/ActivationFunctions.h>
#include <OpenANN/Regularization.h>

namespace OpenANN
{

/**
 * @class Convolutional
 *
 * Applies a learnable filter on a 2D or 3D input.
 *
 * Hence, convolutional layers can be regarded as biologically inspired
 * trainable feature extractors. Another perspective is that they combine
 * weight sharing and sparse connections to reduce the number of weights
 * drastically in contrast to fully connected layers.
 *
 * Each feature map in this layer is connected to each feature map in the
 * previous layer such that we use one convolution kernel for each of these
 * connections. After convolving the input feature maps, an activation
 * function will be applied on the activations.
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
class Convolutional : public Layer
{
  int I, fmin, inRows, inCols, fmout, kernelRows, kernelCols;
  bool bias;
  ActivationFunction act;
  double stdDev;
  Eigen::MatrixXd* x;
  //! output feature maps X input feature maps X kernel rows X kernel cols
  std::vector<std::vector<Eigen::MatrixXd> > W;
  std::vector<std::vector<Eigen::MatrixXd> > Wd;
  //! output feature maps X input feature maps
  Eigen::MatrixXd Wb;
  Eigen::MatrixXd Wbd;
  Eigen::MatrixXd a;
  Eigen::MatrixXd y;
  Eigen::MatrixXd yd;
  Eigen::MatrixXd deltas;
  Eigen::MatrixXd e;
  int fmInSize, outRows, outCols, fmOutSize, maxRow, maxCol;
  Regularization regularization;

public:
  Convolutional(OutputInfo info, int featureMaps, int kernelRows,
                int kernelCols, bool bias, ActivationFunction act,
                double stdDev, Regularization regularization);
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

#endif // OPENANN_LAYERS_CONVOLUTIONAL_H_
