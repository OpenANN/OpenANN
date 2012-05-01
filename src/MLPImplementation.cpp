#include <MLPImplementation.h>
#include <CompressionMatrixFactory.h>
#include <AssertionMacros.h>
#include <ActivationFunctions.h>
#include <io/Logger.h>
#include <Random.h>
#include <cmath>

namespace OpenANN
{

MLPImplementation::MLPImplementation()
  : initialized(false), biased(true)
{
  layerInfos.reserve(5);
}

void MLPImplementation::init()
{
  if(initialized)
    return;

  VL = (int) layerInfos.size();
  WL = VL-1;
  L =  WL-1;
  D = layerInfos[0].nodes;
  F = layerInfos[VL-1].nodes;
  OPENANN_CHECK(VL >= 2);

  weights.resize(WL);
  weightDerivatives.resize(WL);
  parameters.resize(WL);
  parameterDerivatives.resize(WL);
  orthogonalFunctions.resize(WL);

  activations.resize(VL);
  outputs.resize(VL);
  derivatives.resize(VL);
  errors.resize(VL);
  deltas.resize(VL);

  P = 0;
  for(int l = 0; l < VL; l++)
    allocateLayer(l);

  derivatives[0].fill(0.0);
  errors[0].fill(0.0);
  deltas[0].fill(0.0);

  parameterPointers.resize(P);
  parameterVector.resize(P);

  int p = 0;
  for(int l = 0; l < WL; l++)
  {
    initializeParameterPointersLayer(p, l);
    initializeOrthogonalFunctions(l);
  }
  OPENANN_CHECK_EQUALS(p, P);

  initialized = true;
}

void MLPImplementation::allocateLayer(int l)
{
  Logger allocationLogger(Logger::NONE, "allocation");
  allocationLogger << "Allocating layer " << l;
  const int currentUnits = layerInfos[l].nodes;
  const int unitsWithBias = currentUnits+1;
  const int weightLayer = l-1;

  switch(layerInfos[l].type)
  {
    case LayerInfo::INPUT:
    {
      allocationLogger << " (input):\n";
      allocationLogger << "\t- activation vector of dimension " << currentUnits << "\n";
      activations[l] = Vt(currentUnits);
      allocationLogger << "\t- output vector of dimension " << unitsWithBias << "\n";
      outputs[l] = Vt(unitsWithBias);
      outputs[l](currentUnits) = biased ? 1.0 : 0.0;

      derivatives[l] = Vt(currentUnits); // TODO avoid duplicate code
      errors[l] = Vt(layerInfos[l].nodes+1);
      deltas[l] = Vt(currentUnits);
      break;
    }
    case LayerInfo::CONVOLUTIONAL:
    {
      allocationLogger << " (convolutional):\n";
      const int weightRows = layerInfos[l].featureMaps * (layerInfos[l].kernelRows * layerInfos[l].kernelCols + biased);
      const int weightCols = layerInfos[l-1].nodesPerDimension[0];

      allocationLogger << "\t- activation vector of dimension " << currentUnits << "\n";
      activations[l] = Vt(currentUnits);
      allocationLogger << "\t- output vector of dimension " << unitsWithBias << "\n";
      outputs[l] = Vt(unitsWithBias);
      outputs[l](currentUnits) = biased ? 1.0 : 0.0;

      allocationLogger << "\t- weight matrix of dimensions " << weightRows << " x " << weightCols << "\n";
      weights[weightLayer] = Mt(weightRows, weightCols);
      weightDerivatives[weightLayer] = Mt(weightRows, weightCols);

      P += weightRows * weightCols;
      allocationLogger << "\t- set P to " << P << "\n";

      derivatives[l] = Vt(currentUnits);
      errors[l] = Vt(layerInfos[l].nodes+1);
      deltas[l] = Vt(currentUnits);
      break;
    }
    case LayerInfo::FULLY_CONNECTED:
    case LayerInfo::COMPRESSED_FULLY_CONNECTED:
    {
      allocationLogger << " (fully connected):\n";

      allocationLogger << "\t- activation vector of dimension " << currentUnits << "\n";
      activations[l] = Vt(currentUnits);
      allocationLogger << "\t- output vector of dimension " << unitsWithBias << "\n";
      outputs[l] = Vt(unitsWithBias);
      outputs[l](currentUnits) = biased ? 1.0 : 0.0;

      const int lastLayerUnitsWithBias = layerInfos[l-1].nodes+1;
      const int currentParameters = layerInfos[l].compressed ?
          layerInfos[l].parameters : lastLayerUnitsWithBias - (1-biased);
      allocationLogger << "\t- weight matrix of dimensions " << currentUnits << " x " << lastLayerUnitsWithBias << "\n";
      weights[weightLayer] = Mt(currentUnits, lastLayerUnitsWithBias);
      weightDerivatives[weightLayer] = Mt(currentUnits, lastLayerUnitsWithBias);
      allocationLogger << "\t- parameter matrix of dimensions " << currentUnits << " x " << currentParameters << "\n";
      parameters[weightLayer] = Mt(currentUnits, currentParameters);
      parameterDerivatives[weightLayer] = Mt(currentUnits, currentParameters);
      orthogonalFunctions[weightLayer] = Mt(currentParameters, lastLayerUnitsWithBias);

      P += currentUnits * currentParameters;
      allocationLogger << "\t- set P to " << P << "\n";

      derivatives[l] = Vt(currentUnits);
      errors[l] = Vt(layerInfos[l].nodes+1);
      deltas[l] = Vt(currentUnits);
      break;
    }
    case LayerInfo::OUTPUT:
    case LayerInfo::COMPRESSED_OUTPUT:
    {
      allocationLogger << " (output):\n";
      activations[l] = Vt(F);
      allocationLogger << "\t- activation and output vectors of dimension " << F << "\n";
      outputs[l] = Vt(F);

      const int lastLayerUnitsWithBias = layerInfos[l-1].nodes+1;
      const int currentParameters = layerInfos[l].compressed ?
          layerInfos[l].parameters : lastLayerUnitsWithBias - (1-biased);
      allocationLogger << "\t- weight matrix of dimensions " << currentUnits << " x " << lastLayerUnitsWithBias << "\n";
      weights[weightLayer] = Mt(currentUnits, lastLayerUnitsWithBias);
      weightDerivatives[weightLayer] = Mt(currentUnits, lastLayerUnitsWithBias);
      allocationLogger << "\t- parameter matrix of dimensions " << currentUnits << " x " << currentParameters << "\n";
      parameters[weightLayer] = Mt(currentUnits, currentParameters);
      parameterDerivatives[weightLayer] = Mt(currentUnits, currentParameters);
      orthogonalFunctions[weightLayer] = Mt(currentParameters, lastLayerUnitsWithBias);

      P += currentUnits * currentParameters;
      allocationLogger << "\t- set P to " << P << "\n";

      derivatives[l] = Vt(currentUnits);
      errors[l] = Vt(layerInfos[l].nodes+1);
      deltas[l] = Vt(currentUnits);
      break;
    }
    default:
      allocationLogger << " (unknown):\n";
      OPENANN_CHECK(false && "Unknown layer type in allocation.");
      break;
  }
}

void MLPImplementation::initializeParameterPointersLayer(int& p, int l)
{
  switch(layerInfos[l+1].type)
  {
    case LayerInfo::CONVOLUTIONAL:
    case LayerInfo::FULLY_CONNECTED:
    case LayerInfo::OUTPUT:
    {
      for(int i = 0; i < weights[l].cols()-(1-biased); i++)
      {
        for(int j = 0; j < weights[l].rows(); j++)
        {
          parameterVector(p) = weights[l](j, i);
          parameterPointers[p++] = &weights[l](j, i);
        }
      }
      break;
    }
    case LayerInfo::COMPRESSED_FULLY_CONNECTED:
    case LayerInfo::COMPRESSED_OUTPUT:
    {
      OPENANN_CHECK_EQUALS(parameters[l].rows(), layerInfos[l+1].nodes);
      OPENANN_CHECK_EQUALS(parameters[l].cols(), orthogonalFunctions[l].rows());
      OPENANN_CHECK_EQUALS(orthogonalFunctions[l].cols(), layerInfos[l].nodes+1);
      for(int j = 0; j < parameters[l].rows(); j++)
      {
        for(int m = 0; m < parameters[l].cols(); m++)
        {
          parameterVector(p) = parameters[l](j, m);
          parameterPointers[p++] = &parameters[l](j, m);
        }
      }
      break;
    }
    default:
      OPENANN_CHECK(false && "Unkown layer type in initializeParameterPointersLayer.");
      break;
  }
}

void MLPImplementation::initializeOrthogonalFunctions(int l)
{
  CompressionMatrixFactory compressionMatrixFactory;
  compressionMatrixFactory.inputDim = layerInfos[l].nodes+1;
  if(layerInfos[l+1].compression == "random")
    compressionMatrixFactory.transformation = CompressionMatrixFactory::GAUSSIAN;
  else
    compressionMatrixFactory.transformation = CompressionMatrixFactory::DCT;
  if(layerInfos[l+1].compressed)
    compressionMatrixFactory.paramDim = layerInfos[l+1].parameters;
  else
  {
    compressionMatrixFactory.compress = false;
    compressionMatrixFactory.paramDim = layerInfos[l].nodes + biased;
  }
  compressionMatrixFactory.createCompressionMatrix(orthogonalFunctions[l]);
}

void MLPImplementation::initialize()
{
  OPENANN_CHECK(initialized);
  for(int l = 1; l < VL; l++)
    initializeLayer(l);
  for(int p = 0; p < P; p++)
    parameterVector(p) = *parameterPointers[p];
  generateWeightsFromParameters();
}

void MLPImplementation::initializeLayer(int l)
{
  RandomNumberGenerator rng;
  const int previousLayer = l-1;
  switch(layerInfos[l].type)
  {
    case LayerInfo::INPUT: // do nothing
      break;
    case LayerInfo::COMPRESSED_FULLY_CONNECTED:
    case LayerInfo::COMPRESSED_OUTPUT:
    {
      for(int j = 0; j < parameters[previousLayer].rows(); j++)
      {
        for(int m = 0; m < parameters[previousLayer].cols(); m++)
        {
          fpt sigma = 0.333*pow((fpt) m / (fpt) parameters[previousLayer].cols() + 0.333, 1.05);
          parameters[previousLayer](j, m) = rng.sampleNormalDistribution<fpt>() * sigma;
        }
      }
      break;
    }
    case LayerInfo::CONVOLUTIONAL:
    case LayerInfo::FULLY_CONNECTED:
    case LayerInfo::OUTPUT:
    {
      for(int i = 0; i < weights[previousLayer].cols(); i++)
        for(int j = 0; j < weights[previousLayer].rows(); j++)
          weights[previousLayer](j, i) = rng.sampleNormalDistribution<fpt>() * 0.05;
      break;
    }
    default:
      OPENANN_CHECK(false && "Initialiation is not defined for this layer type.");
      break;
  }
}

void MLPImplementation::generateWeightsFromParameters()
{
  for(int l = 0; l < WL; l++)
    if(layerInfos[l+1].compressed)
      weights[l] = parameters[l] * orthogonalFunctions[l];
}

void MLPImplementation::constantParameters(fpt c)
{
  OPENANN_CHECK(initialized);
  Vt constant(P);
  constant.fill(c);
  set(constant);
}

Vt MLPImplementation::operator()(const Vt& x)
{
  OPENANN_CHECK(initialized);

  OPENANN_CHECK_EQUALS(x.rows(), D);
  OPENANN_CHECK_EQUALS(D, activations[0].rows());
  activations[0] = x;

  for(int l = 0; l < VL; l++)
  {
    forwardLayer(l);
    outputLayer(l);
  }

  return outputs[WL];
}

void MLPImplementation::forwardLayer(int l)
{
  switch(layerInfos[l].type)
  {
    case LayerInfo::INPUT:
      break;
    case LayerInfo::CONVOLUTIONAL:
    {
      const int previousLayer = l-1;
      for(int i = 0; i < layerInfos[l].nodes; i++)
        activations[l](i) = 0.0;

      const int arrays = layerInfos[previousLayer].nodesPerDimension[0];
      const int arrayLength = layerInfos[previousLayer].nodesPerDimension[1] * layerInfos[previousLayer].nodesPerDimension[2];
      const int inputRowLenght = layerInfos[previousLayer].nodesPerDimension[1];
      const int inputColLength = layerInfos[previousLayer].nodesPerDimension[2];
      const int featureMapParameters = layerInfos[l].kernelCols * layerInfos[l].kernelRows + biased;
      const int featureMapLenght = layerInfos[l].nodesPerDimension[1] * layerInfos[l].nodesPerDimension[2];
      const int halfKernelRows = layerInfos[l].kernelRows/2;
      const int maxRow = inputRowLenght-halfKernelRows;
      const int halfKernelCols = layerInfos[l].kernelCols/2;
      const int maxCol = inputColLength-halfKernelCols;
      const int kernelRows = layerInfos[l].kernelRows;
      const int kernelCols = layerInfos[l].kernelCols;
      for(int fm = 0, featureMapOutputOffset = 0, featureMapWeightOffset = 0;
          fm < layerInfos[l].featureMaps;
          fm++, featureMapOutputOffset+=featureMapLenght,
            featureMapWeightOffset+=featureMapParameters)
      {
        for(int array = 0, inputIdxOffset1 = 0; array < arrays; array++, inputIdxOffset1+=arrayLength)
        {
          int outputIdx = featureMapOutputOffset;
          for(int row = halfKernelRows; row <= maxRow; row+=2)
          {
            for(int col = halfKernelCols; col <= maxCol; col+=2, outputIdx++)
            {
              int weightOffset = featureMapWeightOffset;
              for(int kr = 0, inputIdxOffset2 = inputIdxOffset1 + (row-halfKernelRows)*inputColLength; kr < kernelRows; kr++, inputIdxOffset2+=inputColLength)
              {
                for(int kc = 0, inputIdx = inputIdxOffset2+col-halfKernelCols; kc < kernelCols; kc++, inputIdx++)
                {
                  OPENANN_CHECK_WITHIN(outputIdx, 0, layerInfos[l].nodes-1);
                  activations[l](outputIdx) += weights[previousLayer](weightOffset++, array) * outputs[previousLayer][inputIdx];
                }
              }
              if(biased)
                activations[l](outputIdx) += weights[previousLayer](weightOffset, array);
            }
          }
        }
      }
      break;
    }
    case LayerInfo::FULLY_CONNECTED:
    case LayerInfo::COMPRESSED_FULLY_CONNECTED:
    case LayerInfo::OUTPUT:
    case LayerInfo::COMPRESSED_OUTPUT:
    {
      const int previousLayer = l-1;
      OPENANN_CHECK_EQUALS(outputs[previousLayer].rows(), weights[previousLayer].cols());
      OPENANN_CHECK_EQUALS(weights[previousLayer].rows(), activations[l].rows());
      activations[l] = weights[previousLayer] * outputs[previousLayer];
      break;
    }
    default:
      OPENANN_CHECK(false && "Unkown layer type in forward propagation.");
      break;
  }
}

void MLPImplementation::outputLayer(int l)
{
  switch(layerInfos[l].type)
  {
    case LayerInfo::INPUT:
    {
      OPENANN_CHECK_EQUALS(D+1, outputs[l].rows());
      for(int d = 0; d < D; d++)
        outputs[l](d) = activations[l](d);
      break;
    }
    case LayerInfo::CONVOLUTIONAL:
    case LayerInfo::FULLY_CONNECTED:
    case LayerInfo::COMPRESSED_FULLY_CONNECTED:
    case LayerInfo::OUTPUT:
    case LayerInfo::COMPRESSED_OUTPUT:
    {
      OPENANN_CHECK_EQUALS((l != WL ? 1 : 0) + activations[l].rows(), outputs[l].rows());
      OPENANN_CHECK_EQUALS(layerInfos[l].nodes, activations[l].rows());
      switch(layerInfos[l].a)
      {
        case SIGMOID:
          OpenANN::logistic(activations[l], outputs[l]);
          break;
        case TANH:
          OpenANN::normaltanh(activations[l], outputs[l]);
          break;
        case ID:
          OpenANN::linear(activations[l], outputs[l]);
          break;
        default:
          OPENANN_CHECK(false && "Unknow activation function.");
          break;
      }
      break;
    }
    default:
      OPENANN_CHECK(false && "Unkown layer type in forward propagation.");
      break;
  }
}

void MLPImplementation::backpropagate(const Vt& t)
{
  OPENANN_CHECK(initialized);
  OPENANN_CHECK_EQUALS(VL, (int) deltas.size());
  OPENANN_CHECK_EQUALS(VL, (int) derivatives.size());

  OPENANN_CHECK_EQUALS(F, t.rows());
  errors[WL] = outputs[WL] - t;

  for(int l = 1; l < VL; l++)
    calculateDerivativesLayer(l);

  for(int l = WL; l > 0; l--)
    backpropDeltasLayer(l);

  for(int l = 0; l < WL; l++)
    calculateGradientLayer(l);
}

void MLPImplementation::calculateDerivativesLayer(int l)
{
  OPENANN_CHECK_EQUALS(derivatives[l].rows(), activations[l].rows());
  switch(layerInfos[l].a)
  {
    case SIGMOID:
      OpenANN::logisticDerivative(outputs[l], derivatives[l]);
      break;
    case TANH:
      OpenANN::normaltanhDerivative(outputs[l], derivatives[l]);
      break;
    case ID:
      OpenANN::linearDerivative(derivatives[l]);
      break;
    default:
      OPENANN_CHECK(false && "Unknow activation function.");
      break;
  }
}

void MLPImplementation::backpropDeltasLayer(int l)
{
  const int previousLayer = l-1;
  switch(layerInfos[l].type)
  {
    case LayerInfo::INPUT: // do nothing
      break;
    case LayerInfo::CONVOLUTIONAL:
    {
      for(int i = 0; i < layerInfos[l].nodes; i++)
        deltas[l](i) = derivatives[l](i) * errors[l](i);

      errors[previousLayer].fill(0.0);

      const int arrayLength = layerInfos[previousLayer].nodesPerDimension[1] * layerInfos[previousLayer].nodesPerDimension[2];
      const int inputRowLenght = layerInfos[previousLayer].nodesPerDimension[1];
      const int inputColLength = layerInfos[previousLayer].nodesPerDimension[2];
      const int featureMapParameters = layerInfos[l].kernelCols * layerInfos[l].kernelRows + biased;
      const int featureMapLenght = layerInfos[l].nodesPerDimension[1] * layerInfos[l].nodesPerDimension[2];
      const int halfKernelRows = layerInfos[l].kernelRows/2;
      const int halfKernelCols = layerInfos[l].kernelCols/2;
      for(int fm = 0; fm < layerInfos[l].featureMaps; fm++)
      {
        for(int array = 0; array < layerInfos[previousLayer].nodesPerDimension[0]; array++)
        {
          int inputOffset = 0;
          for(int row = halfKernelRows; row <= inputRowLenght-halfKernelRows; row+=2)
          {
            for(int col = halfKernelCols; col <= inputColLength-halfKernelCols; col+=2)
            {
              int weightOffset = 0;
              for(int kr = -halfKernelRows; kr <= halfKernelRows; kr++)
              {
                for(int kc = -halfKernelCols; kc <= halfKernelCols; kc++)
                {
                  const int currentInputRow = row+kr;
                  const int currentInputCol = col+kc;
                  const int inputIdx = array*arrayLength + currentInputRow*inputColLength + currentInputCol;
                  const int outputIdx = fm*featureMapLenght + inputOffset;
                  OPENANN_CHECK(outputIdx < layerInfos[l].nodes);
                  const int weightIdx = fm*featureMapParameters + weightOffset++;
                  if(inputIdx < errors[previousLayer].rows())
                    errors[previousLayer](inputIdx) += weights[previousLayer](weightIdx, array) * deltas[l](outputIdx);
                }
              }
              inputOffset++;
            }
          }
        }
      }
      break;
    }
    case LayerInfo::FULLY_CONNECTED:
    case LayerInfo::COMPRESSED_FULLY_CONNECTED:
    {
      OPENANN_CHECK_EQUALS(weights[previousLayer].cols(), errors[previousLayer].rows());
      OPENANN_CHECK_EQUALS(weights[previousLayer].rows(), deltas[l].rows());
      OPENANN_CHECK_EQUALS(errors[l].rows(), derivatives[l].rows()+1);
      for(int j = 0; j < layerInfos[l].nodes; j++)
        deltas[l](j) = derivatives[l](j) * errors[l](j);
      errors[previousLayer] = weights[previousLayer].transpose() * deltas[l];
      break;
    }
    case LayerInfo::OUTPUT:
    case LayerInfo::COMPRESSED_OUTPUT:
    {
      OPENANN_CHECK_EQUALS(weights[previousLayer].cols(), errors[previousLayer].rows());
      OPENANN_CHECK_EQUALS(weights[previousLayer].rows(), deltas[l].rows());
      OPENANN_CHECK_EQUALS(errors[l].rows(), derivatives[l].rows());
      deltas[l] = derivatives[l].asDiagonal() * errors[l];
      errors[previousLayer] = weights[previousLayer].transpose() * deltas[l];
      break;
    }
    default:
      OPENANN_CHECK(false && "Unknown layer type for backpropagation of deltas.");
      break;
  }
}

void MLPImplementation::calculateGradientLayer(int l)
{
  OPENANN_CHECK_EQUALS(outputs[l].rows(), layerInfos[l].nodes+1);
  OPENANN_CHECK_EQUALS(deltas[l+1].rows(), layerInfos[l+1].nodes);

  if(layerInfos[l+1].type == LayerInfo::CONVOLUTIONAL)
  {
    weightDerivatives[l].fill(0.0);

    const int arrayLength = layerInfos[l].nodesPerDimension[1] * layerInfos[l].nodesPerDimension[2];
    const int inputRowLenght = layerInfos[l].nodesPerDimension[1];
    const int inputColLength = layerInfos[l].nodesPerDimension[2];
    const int featureMapParameters = layerInfos[l+1].kernelCols * layerInfos[l+1].kernelRows + biased;
    const int featureMapLenght = layerInfos[l+1].nodesPerDimension[1] * layerInfos[l+1].nodesPerDimension[2];
    const int halfKernelRows = layerInfos[l+1].kernelRows/2;
    const int halfKernelCols = layerInfos[l+1].kernelCols/2;
    for(int fm = 0; fm < layerInfos[l+1].featureMaps; fm++)
    {
      for(int array = 0; array < layerInfos[l].nodesPerDimension[0]; array++)
      {
        int inputOffset = 0;
        for(int row = halfKernelRows; row <= inputRowLenght-halfKernelRows; row+=2)
        {
          for(int col = halfKernelCols; col <= inputColLength-halfKernelCols; col+=2)
          {
            const int outputIdx = fm*featureMapLenght + inputOffset;
            int weightOffset = fm*featureMapParameters;
            for(int kr = -halfKernelRows; kr <= halfKernelRows; kr++)
            {
              for(int kc = -halfKernelCols; kc <= halfKernelCols; kc++)
              {
                const int currentInputRow = row+kr;
                const int currentInputCol = col+kc;
                const int inputIdx = array*arrayLength + currentInputRow*inputColLength + currentInputCol;
                OPENANN_CHECK(outputIdx < layerInfos[l+1].nodes);
                if(inputIdx < outputs[l].rows())
                  weightDerivatives[l](weightOffset++, array) += deltas[l+1](outputIdx)*outputs[l](inputIdx);
              }
            }
            weightDerivatives[l](weightOffset++, array) += deltas[l+1](outputIdx);
            inputOffset++;
          }
        }
      }
    }
  }
  else
  {
    OPENANN_CHECK_EQUALS(weightDerivatives[l].rows(), layerInfos[l+1].nodes);
    OPENANN_CHECK_EQUALS(weightDerivatives[l].cols(), layerInfos[l].nodes+1);
    weightDerivatives[l] = deltas[l+1] * outputs[l].transpose();
    if(layerInfos[l+1].compressed)
    {
      OPENANN_CHECK_EQUALS(parameterDerivatives[l].rows(), layerInfos[l+1].nodes);
      OPENANN_CHECK_EQUALS(parameterDerivatives[l].cols(), layerInfos[l+1].parameters);
      OPENANN_CHECK_EQUALS(orthogonalFunctions[l].rows(), layerInfos[l+1].parameters);
      OPENANN_CHECK_EQUALS(orthogonalFunctions[l].cols(), layerInfos[l].nodes+1);
      OPENANN_CHECK_EQUALS(weightDerivatives[l].rows(), layerInfos[l+1].nodes);
      OPENANN_CHECK_EQUALS(weightDerivatives[l].cols(), layerInfos[l].nodes+1);
      parameterDerivatives[l] = weightDerivatives[l] * orthogonalFunctions[l].transpose();
    }
  }
}

void MLPImplementation::derivative(Vt& g)
{
  OPENANN_CHECK(initialized);
  OPENANN_CHECK(P == g.rows());
  int p = 0;
  for(int l = 0; l < WL; l++)
    derivativeLayer(g, p, l);
  OPENANN_CHECK(P == p);
}

void MLPImplementation::derivativeLayer(Vt& g, int& p, int l)
{
  if(layerInfos[l+1].compressed)
  {
    for(int j = 0; j < parameterDerivatives[l].rows(); j++)
      for(int m = 0; m < parameterDerivatives[l].cols(); m++)
        g(p++) += parameterDerivatives[l](j, m);
  }
  else
  {
    for(int i = 0; i < weightDerivatives[l].cols()-(1-biased); i++)
      for(int j = 0; j < weightDerivatives[l].rows(); j++)
        g(p++) += weightDerivatives[l](j, i);
  }
}

void MLPImplementation::singleDerivative(Vt& g)
{
  OPENANN_CHECK(initialized);
  OPENANN_CHECK(P == g.rows());
  int p = 0;
  for(int l = 0; l < WL; l++)
    singleDerivativeLayer(g, p, l);
  OPENANN_CHECK(P == p);
}

void MLPImplementation::singleDerivativeLayer(Vt& g, int& p, int l)
{
  if(layerInfos[l+1].compressed)
  {
    for(int j = 0; j < parameterDerivatives[l].rows(); j++)
      for(int m = 0; m < parameterDerivatives[l].cols(); m++)
        g(p++) = parameterDerivatives[l](j, m);
  }
  else
  {
    for(int i = 0; i < weightDerivatives[l].cols()-(1-biased); i++)
      for(int j = 0; j < weightDerivatives[l].rows(); j++)
        g(p++) = weightDerivatives[l](j, i);
  }
}

void MLPImplementation::set(const Vt& newParameters)
{
  OPENANN_CHECK(initialized);
  parameterVector = newParameters;
  OPENANN_CHECK_EQUALS(P, newParameters.rows());
  for(int p = 0; p < P; p++)
    *parameterPointers[p] = newParameters(p);
  generateWeightsFromParameters();
}

const Vt& MLPImplementation::get()
{
  return parameterVector;
}

}
