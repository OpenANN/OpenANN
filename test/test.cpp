#include "Test/TestSuite.h"
#include "Test/TextTestRunner.h"
#ifdef USE_QT
#include "Test/QtTestRunner.h"
#endif
#include <iostream>
#include <cstdlib>
#include <io/Logger.h>
#include <OpenANN>
#include "ActivationFunctionsTestCase.h"
#include "CMAESTestCase.h"
#include "CompressionMatrixFactoryTestCase.h"
#include "MLPImplementationTestCase.h"
#include "MLPTestCase.h"
#include "RandomTestCase.h"
#include "LayerTestCase.h"

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

  TestSuite ts("OpenANN");
  //ts.addTestCase(new ActivationFunctionsTestCase);
  //ts.addTestCase(new CMAESTestCase);
  //ts.addTestCase(new CompressionMatrixFactoryTestCase);
  //ts.addTestCase(new MLPImplementationTestCase);
  //ts.addTestCase(new MLPTestCase);
  //ts.addTestCase(new RandomTestCase);
  ts.addTestCase(new LayerTestCase);

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
