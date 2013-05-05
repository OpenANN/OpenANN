#pragma once

#include <Test/TestCase.h>
#include <Eigen/Dense>

class DataSetTestCase : public TestCase
{
public:
  virtual void run();

  void directStorageDataSets();
  void dataSetViews();
  void dataSetSplitsFromGroups();
  void dataSetSplitsFromRatio();
  void dataSetMerge();
};

