#pragma once

#include <Eigen/Dense>
#include <vector>

namespace OpenANN
{

class MLPImplementation
{
public:
  enum ActivationFunction
  {
    SIGMOID = 0,
    TANH = 1,
    ID = 2,
    STANH = 3
  };

  struct LayerInfo
  {
    enum Type
    {
      INPUT,
      CONVOLUTIONAL,
      FULLY_CONNECTED,
      COMPRESSED_FULLY_CONNECTED,
      OUTPUT,
      COMPRESSED_OUTPUT
    } type;
    int dimension;
    std::vector<int> nodesPerDimension;
    int nodes;
    bool bias;
    ActivationFunction a;
    bool compressed;
    int parameters;
    std::string compression;
    int featureMaps;
    int kernelRows;
    int kernelCols;

    LayerInfo(Type type, int dimension, int nodes, ActivationFunction a)
      : type(type), dimension(dimension), nodes(nodes), bias(true), a(a),
        compressed(false), parameters(-1), compression("dct"), featureMaps(-1),
        kernelRows(-1), kernelCols(-1)
    {
    }

    void compress(int parameters, std::string compression = "dct")
    {
      compressed = true;
      this->parameters = parameters;
      this->compression = compression;
      switch(type)
      {
        case FULLY_CONNECTED:
          type = COMPRESSED_FULLY_CONNECTED;
          break;
        case OUTPUT:
          type = COMPRESSED_OUTPUT;
          break;
        default:
          break;
      }
    }
  };

  std::vector<LayerInfo> layerInfos;
  int P; //! number of parameters

  // Network topology
  bool initialized;
  bool biased;
  int D; //!< input dimension
  int F; //!< output dimension
  int L; //!< hidden layers
  std::vector<Mt> weights;
  int WL; //!< weight layers; output and between each hidden layer
  int VL; //!< value layers; input, output and between each hidden layer

  // Compression
  std::vector<Mt> parameters;
  std::vector<Mt> orthogonalFunctions;
  std::vector<fpt*> parameterPointers;
  Vt parameterVector;

  // Forward propagation
  std::vector<Vt> activations;
  std::vector<Vt> outputs;

  // Backpropagation
  std::vector<Vt> derivatives;
  std::vector<Vt> errors;
  std::vector<Vt> deltas;
  std::vector<Mt> weightDerivatives;

  // Compressed backpropagation
  std::vector<Mt> parameterDerivatives;

  // N-dimensional first layer
  std::vector<std::vector<fpt> > firstLayer3Dt;
  int parametersX;
  int parametersY;
  int parametersZ;

  MLPImplementation();
  void init();
  void allocateLayer(int l);
  void initializeParameterPointersLayer(int& p, int l);
  void initializeOrthogonalFunctions(int l);
  void initialize();
  void initializeLayer(int l);
  void generateWeightsFromParameters();
  void constantParameters(fpt c);
  Vt operator()(const Vt& x);
  void forwardLayer(int l);
  void outputLayer(int l);
  void calculateDerivativesLayer(int l);
  void backpropagate(const Vt& t);
  void backpropDeltasLayer(int l);
  void calculateGradientLayer(int l);
  void derivative(Vt& g);
  void derivativeLayer(Vt& g, int& p, int l);
  void singleDerivative(Vt& g);
  void singleDerivativeLayer(Vt& g, int& p, int l);
  void set(const Vt& newParameters);
  const Vt& get();
};

}
