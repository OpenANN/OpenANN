#pragma once

#include <OpenANN>
#include <CompressionMatrixFactory.h>
#include <fstream>

using namespace OpenANN;

class MLPWrapper
{
  MLP mlp;
  Mt trainingInput, traininOutput, testInput, testOutput;

  MLP::ErrorFunction e(std::string err)
  {
    if(err == "sse")
      return MLP::SSE;
    else if(err == "ce")
      return MLP::CE;
    else
      return MLP::MSE;
  }

  MLP::ActivationFunction g(std::string act)
  {
    if(act == "sigmoid")
      return MLP::SIGMOID;
    else if(act == "id")
      return MLP::ID;
    else if(act == "sm")
      return MLP::SM;
    else
      return MLP::TANH;
  }

public:
  MLPWrapper() : mlp(Logger::CONSOLE, Logger::NONE)
  {
  }

  void noBias()
  {
    mlp.noBias();
  }

  void input(int units)
  {
    mlp.input(units);
  }

  void input2D(int unitsX, int unitsY)
  {
    mlp.input(unitsX, unitsY);
  }

  void input3D(int unitsX, int unitsY, int unitsZ)
  {
    mlp.input(unitsX, unitsY, unitsZ);
  }

  void fullyConnectedHiddenLayer(int units, std::string act, int parameters)
  {
    mlp.fullyConnectedHiddenLayer(units, g(act), parameters);
  }

  void convolutionalLayer(int featureMaps, int kernelRows, int kernelCols, std::string act)
  {
    mlp.convolutionalLayer(featureMaps, kernelRows, kernelCols, g(act));
  }

  void output(int units, std::string err, std::string act, int parameters)
  {
    mlp.output(units, e(err), g(act), parameters);
  }

  void trainingSet(const Mt in, const Mt out)
  {
    trainingInput = in;
    traininOutput = out;
    mlp.trainingSet(trainingInput, traininOutput);
  }

  void testSet(const Mt in, const Mt out)
  {
    testInput = in;
    testOutput = out;
    mlp.testSet(testInput, testOutput);
  }

  void training(std::string algorithm)
  {
    if(algorithm == "lma")
      mlp.training(MLP::BATCH_LMA);
    else if(algorithm == "sgd")
      mlp.training(MLP::BATCH_SGD);
    else if(algorithm == "cmaes")
      mlp.training(MLP::BATCH_CMAES);
  }

  void fitToError(fpt minimalValue)
  {
    StopCriteria stop;
    stop.minimalValue = minimalValue;
    mlp.fit(stop);
  }

  void fitToDiff(fpt minimalStep)
  {
    StopCriteria stop;
    stop.minimalValueDifferences = minimalStep;
    stop.minimalSearchSpaceStep = minimalStep;
    mlp.fit(stop);
  }

  Vt value(Vt in)
  {
    return mlp(in);
  }

  Vt getParameters()
  {
    return mlp.currentParameters();
  }

  void setParameters(Vt parameters)
  {
    mlp.setParameters(parameters);
  }
};

class CompressionMatrixFactoryWrapper
{
  CompressionMatrixFactory cmf;
  Mt cm;

  CompressionMatrixFactory::Transformation transformation(std::string t)
  {
    if(t == "avg")
      return CompressionMatrixFactory::AVERAGE;
    else if(t == "edge")
      return CompressionMatrixFactory::EDGE;
    else if(t == "gauss")
      return CompressionMatrixFactory::GAUSSIAN;
    else if(t == "sparse")
      return CompressionMatrixFactory::SPARSE_RANDOM;
    else
      return CompressionMatrixFactory::DCT;
  }

public:

  CompressionMatrixFactoryWrapper(int firstInputDim, int firstParamDim,
      std::string t = "dct")
    : cmf(firstInputDim, firstParamDim, transformation(t)), cm(firstParamDim, firstInputDim)
  {
    cm.fill(0.0);
    cmf.createCompressionMatrix(cm);
  }

  void readFromFile(std::string fileName)
  {
    std::fstream file(fileName.c_str());
    fpt tmp = 0.0;
    for(int i = 0; i < cm.rows(); i++)
    {
      for(int j = 0; j < cm.cols(); j++)
      {
        file >> tmp;
        cm(i, j) = tmp;
      }
    }
  }

  Vt compress(Vt uncompressed)
  {
    return cm * uncompressed;
  }
};
