#include "IODataSetTestCase.h"
#include <OpenANN/io/LibSVM.h>
#include <sstream>

void IODataSetTestCase::run()
{
  RUN(IODataSetTestCase, loadLibSVM);
  RUN(IODataSetTestCase, saveLibSVM);
}


void IODataSetTestCase::loadLibSVM()
{
  std::string data = 
    "1 1:2.5 2:2.1\n"
    "0 2:3.1 3:0.5\n"
    "1 1:0.1 2:0.2 3:0.4\n";

  std::stringstream str(data);

  Eigen::MatrixXd input;
  Eigen::MatrixXd output;

  ASSERT(OpenANN::LibSVM::load(input, output, str) > 0);

  Eigen::MatrixXd X(3, 3);
  X << 2.5, 2.1, 0.0, 
       0.0, 3.1, 0.5,
       0.1, 0.2, 0.4;

  Eigen::MatrixXd Y(3, 1);
  Y << 1.0, 0.0, 1.0;

  ASSERT_EQUALS(X, input);
  ASSERT_EQUALS(Y, output);
}


void IODataSetTestCase::saveLibSVM()
{
  std::stringstream str;

  Eigen::MatrixXd input = Eigen::MatrixXd::Identity(2, 2) * 1.5;

  Eigen::MatrixXd output(2, 1);
  output << 1.0, 0.0;

  OpenANN::LibSVM::save(input, output, str);

  ASSERT_EQUALS(str.str(), 
      "1 1:1.5\n0 2:1.5\n");
}

