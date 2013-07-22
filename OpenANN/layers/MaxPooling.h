#ifndef OPENANN_LAYERS_MAX_POOLING_H_
#define OPENANN_LAYERS_MAX_POOLING_H_

#include <OpenANN/layers/Layer.h>
#include <OpenANN/ActivationFunctions.h>

namespace OpenANN
{

/**
 * @class MaxPooling
 *
 * Performs max-pooling on 2D input feature maps.
 *
 * In comparison to average pooling this we have no weights or biases and no
 * activation functions in a max-pooling layer. Instead of summing the inputs
 * up, we only take the maximum value. Max-pooling layer are usually more
 * efficient than subsampling layers and achieve better results.
 *
 * [1] D. Scherer, A. Müller and S. Behnke:
 * Evaluation of Pooling Operations in Convolutional Architectures for Object
 * Recognition.
 * International Conference on Artificial Neural Networks, 2010.
 */
class MaxPooling : public Layer
{
  int I, fm, inRows, inCols, kernelRows, kernelCols;
  Eigen::MatrixXd* x;
  Eigen::MatrixXd y;
  Eigen::MatrixXd deltas;
  Eigen::MatrixXd e;
  int fmInSize, outRows, outCols, fmOutSize, maxRow, maxCol;

public:
  MaxPooling(OutputInfo info, int kernelRows, int kernelCols);
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

#endif // OPENANN_LAYERS_MAX_POOLING_H_
