#pragma once

#include <Test/TestCase.h>
#include <Eigen/Dense>

class DataSetTestCase : public TestCase
{
public:
  DataSetTestCase();
  virtual ~DataSetTestCase() {}

  virtual void run();

  void directStorageDataSets();
  void dataSetViews();
  void dataSetSplitsFromGroups();
  void dataSetSplitsFromRatio();
  void dataSetMerge();

private:
  Eigen::MatrixXd in;
  Eigen::MatrixXd out;
};

