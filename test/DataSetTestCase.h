#pragma once

#include <Test/TestCase.h>
#include <Eigen/Dense>

class DataSetTestCase : public TestCase
{
public:
  DataSetTestCase();
  virtual ~DataSetTestCase() {}

  virtual void run();

  void DirectStorageDataSets();
  void DataSetViews();
  void DataSetSplitsFromGroups();
  void DataSetSplitsFromRatio();
  void DataSetMerge();

private:
  Eigen::MatrixXd in;
  Eigen::MatrixXd out;
};

