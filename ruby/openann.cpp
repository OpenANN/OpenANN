#include "rice/Data_Type.hpp"
#include "rice/Constructor.hpp"
#include "rice/Array.hpp"
#include "MLPWrapper.h"

using namespace Rice;

template<>
Mt from_ruby<Mt>(Object x)
{
  Array array(x);
  Array firstRow(array[0]);
  const size_t rows = array.size();
  const size_t cols = firstRow.size();
  Mt res(cols, rows);
  for(size_t m = 0; m < rows; m++)
  {
    Array row(array[m]);
    for(size_t n = 0; n < cols; n++)
    {
      res(n, m) = from_ruby<fpt>(row[n].value());
    }
  }
  return res;
}

template<>
Vt from_ruby<Vt>(Object x)
{
  Array array(x);
  const size_t rows = array.size();
  Vt res(rows);
  for(size_t m = 0; m < rows; m++)
    res(m) = from_ruby<fpt>(array[m].value());
  return res;
}

template<>
Object to_ruby<Vt>(const Vt& x)
{
  Array array;
  for(int i = 0; i < x.rows(); i++)
    array.push(to_ruby(x(i)));
  return array;
}

extern "C"
void Init_openann()
{
  Data_Type<MLPWrapper> rb_cMLPWrapper = define_class<MLPWrapper>("MLP")
      .define_constructor(Constructor<MLPWrapper>())
      .define_method("no_bias", &MLPWrapper::noBias)
      .define_method("input", &MLPWrapper::input)
      .define_method("input2D", &MLPWrapper::input2D)
      .define_method("input3D", &MLPWrapper::input3D)
      .define_method("fully_connected_hidden_layer", &MLPWrapper::fullyConnectedHiddenLayer)
      .define_method("convolutional_layer", &MLPWrapper::convolutionalLayer)
      .define_method("output", &MLPWrapper::output)
      .define_method("training_set", &MLPWrapper::trainingSet)
      .define_method("test_set", &MLPWrapper::testSet)
      .define_method("training", &MLPWrapper::training)
      .define_method("fit_to_error", &MLPWrapper::fitToError)
      .define_method("fit_to_diff", &MLPWrapper::fitToDiff)
      .define_method("value", &MLPWrapper::value)
      .define_method("parameters", &MLPWrapper::getParameters)
      .define_method("parameters=", &MLPWrapper::setParameters);

  Data_Type<CompressionMatrixFactoryWrapper> rb_cCompression = define_class<CompressionMatrixFactoryWrapper>("Compression")
      .define_constructor(Constructor<CompressionMatrixFactoryWrapper, int, int, std::string>())
      .define_method("read_from_file", &CompressionMatrixFactoryWrapper::readFromFile)
      .define_method("compress", &CompressionMatrixFactoryWrapper::compress);
}
