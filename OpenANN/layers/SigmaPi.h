#ifndef OPENANN_LAYERS_SIGMA_PI_H_
#define OPENANN_LAYERS_SIGMA_PI_H_

#include <Eigen/Dense>
#include <vector>
#include <OpenANN/layers/Layer.h>
#include <OpenANN/ActivationFunctions.h>

namespace OpenANN
{

/**
 * @class SigmaPi
 *
 * Fully connected higher-order layer.
 *
 * For encoding invariances into the topology of the neural network
 * you can specify a weight constraint for a given higher-order node.
 *
 * [1] Max B. Reid, Lilly Spirkovska and Ellen Ochoa
 * Rapid training of higher-order neural network for invariant pattern recognition
 * Proc. IJCNN Int. Conf. Neural Networks, Vol. 1, pp. 689-692, 1989
 *
 * [2] C. L. Gilles and T. Maxwell
 * Learning, invariance, and generalization in high-order neural networks
 * Appl. Opt, Vol. 26, pp. 4972-4978, 1987
 */
class SigmaPi : public Layer
{
protected:
  struct HigherOrderUnit
  {
    std::vector<int> position;
    size_t weight;
  };

  typedef std::vector<HigherOrderUnit> HigherOrderNeuron;

  OutputInfo info;
  bool bias;
  ActivationFunction act;
  double stdDev;

  Eigen::MatrixXd x;
  Eigen::MatrixXd a;
  Eigen::MatrixXd y;
  Eigen::MatrixXd yd;
  Eigen::MatrixXd deltas;
  Eigen::MatrixXd e;

  std::vector<double> w;
  std::vector<double> wd;
  std::vector<HigherOrderNeuron> nodes;

public:
  /**
   * Construct a SigmaPi layer that can be extended with different higher-order nodes
   * @param info OutputInfo of previous, connected layer
   * @param bias flag if this layer supports a bias term for the next, connected layers
   * @param act specifies using activation function for all higher-order nodes
   * @param stdDev defines the standard deviation for the random weight initialization
   */
  SigmaPi(OutputInfo info, bool bias, ActivationFunction act, double stdDev);

  /**
   * See OpenANN::Layer::initialize(std::vector<double*>& pParameter, std::vector<double*>& pDerivative)
   */
  virtual OutputInfo initialize(std::vector<double*>& parameterPointers,
                                std::vector<double*>& parameterDerivativePointers);

  /**
   * A helper class for specifying weight constrains in a higher-order neural network
   * Derive a new class from this interface and simple reimplement the function call operator
   * for the corresponding higher-order term.
   * e.g. constraint(p1, p2, p3) for third-order nodes.
   *
   * NEVER overwrite the isDefault method.
   * (This is only important for unconstrained higher-order nodes)
   */
  struct Constraint
  {
    Constraint() {}
    virtual ~Constraint() {}
    /**
      * function call operator for corresponding second-order nodes
      */
    virtual double operator()(int p1, int p2) const;

    /**
      * function call operator for corresponding third-order nodes
      */
    virtual double operator()(int p1, int p2, int p3) const;

    /**
      * function call operator for corresponding fourth-order nodes
      */
    virtual double operator()(int p1, int p2, int p3, int p4) const;

    /**
      * @internal
      * NEVER overwrite this method.
      */
    virtual bool isDefault() const;
  };

  /**
   * Add a specific number of second-order node to this layer
   * @param numbers number of nodes to add
   * @return return a reference of this instance for convient layer construction
   */
  virtual SigmaPi& secondOrderNodes(int numbers);

  /**
   * Add a specific number of third-order node to this layer
   * @param numbers number of nodes to add
   * @return return a reference of this instance for convient layer construction
   */
  virtual SigmaPi& thirdOrderNodes(int numbers);

  /**
   * Add a specific number of fourth-order node to this layer
   * @param numbers number of nodes to add
   * @return return a reference of this instance for convient layer construction
   */
  virtual SigmaPi& fourthOrderNodes(int numbers);

  /**
   * Add a specific number of second-order nodes that uses the same weight sharing topology
   * @param numbers number of nodes to add
   * @param constrain specifies shared weight groups for signal korrelations from higher-order terms
   * @return return a reference of this instance for convient layer construction
   */
  virtual SigmaPi& secondOrderNodes(int numbers, const Constraint& constrain);

  /**
   * Add a specific number of third-order nodes that uses the same weight sharing topology
   * @param numbers number of nodes to add
   * @param constrain specifies shared weight groups for signal korrelations from higher-order terms
   * @return return a reference of this instance for convient layer construction
   */
  virtual SigmaPi& thirdOrderNodes(int numbers, const Constraint& constrain);

  /**
   * Add a specific number of fourth-order nodes that uses the same weight sharing topology
   * @param numbers number of nodes to add
   * @param constrain specifies shared weight groups for signal korrelations from higher-order terms
   * @return return a reference of this instance for convient layer construction
   */
  virtual SigmaPi& fourthOrderNodes(int numbers, const Constraint& constrain);
  virtual size_t nodenumber() const { return nodes.size(); };
  virtual size_t parameter() const { return w.size(); };
  virtual void initializeParameters();
  virtual void updatedParameters();
  virtual void forwardPropagate(Eigen::MatrixXd* x, Eigen::MatrixXd*& y,
                                bool dropout = false);
  virtual void backpropagate(Eigen::MatrixXd* ein, Eigen::MatrixXd*& eout,
                             bool backpropToPrevious);
  virtual Eigen::MatrixXd& getOutput();
  virtual Eigen::VectorXd getParameters();
};

} // namespace OpenANN


#endif // OPENANN_LAYERS_SIGMA_PI_H_
