#include "Test/TestSuite.h"
#include "Test/TextTestRunner.h"
#ifdef USE_QT
#include "Test/QtTestRunner.h"
#endif
#include <iostream>
#include <cstdlib>
#include <OpenANN/io/Logger.h>
#include <OpenANN/OpenANN>
#include <OpenANN/util/Random.h>
#include "ActivationFunctionsTestCase.h"
#include "CMAESTestCase.h"
#include "CompressionMatrixFactoryTestCase.h"
#include "RandomTestCase.h"
#include "LayerTestCase.h"
#include "PreprocessingTestCase.h"
#include "IntrinsicPlasticityTestCase.h"
#include "RBMTestCase.h"
#include "MBSGDTestCase.h"
#include "LMATestCase.h"
#include "CGTestCase.h"
#include "DataSetTestCase.h"
#include "IODataSetTestCase.h"
#include "EvaluationTestCase.h"

int main(int argc, char** argv)
{
  OpenANN::OpenANNLibraryInfo::print();

  bool verbose = false;
  bool qt = false;
  for(int i = 1; i < argc; i++)
  {
    std::string argument(argv[i]);
    if(argument == std::string("-v"))
      verbose = true;
    else if(argument == std::string("-qt"))
      qt = true;
  }

  OpenANN::Logger::deactivate = true;
  OpenANN::Log::getLevel() = OpenANN::Log::INFO;
  OpenANN::RandomNumberGenerator rng;
  rng.seed(4);

  TestSuite ts("OpenANN");
  ts.addTestCase(new ActivationFunctionsTestCase);
  ts.addTestCase(new CompressionMatrixFactoryTestCase);
  ts.addTestCase(new RandomTestCase);
  ts.addTestCase(new LayerTestCase);
  ts.addTestCase(new PreprocessingTestCase);
  ts.addTestCase(new IntrinsicPlasticityTestCase);
  ts.addTestCase(new RBMTestCase);
  ts.addTestCase(new CMAESTestCase);
  ts.addTestCase(new MBSGDTestCase);
  ts.addTestCase(new LMATestCase);
  ts.addTestCase(new CGTestCase);
  ts.addTestCase(new DataSetTestCase);
  ts.addTestCase(new IODataSetTestCase);
  ts.addTestCase(new EvaluationTestCase);

  if(qt)
  {
#ifdef USE_QT
    QtTestRunner qtr(argc, argv);
    qtr.run(ts);
#else
    std::cerr << "Qt is not available." << std::endl;
    return EXIT_FAILURE;
#endif
  }
  else
  {
    TextTestRunner ttr(verbose);
    ttr.run(ts);
  }

  return EXIT_SUCCESS;
}
