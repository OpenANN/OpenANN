#ifndef OPENANN_TEST_DATA_SET_TEST_CASE_H_
#define OPENANN_TEST_DATA_SET_TEST_CASE_H_

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
  void dataSetSamplingWithoutReplacement();
  void dataSetSamplingWithReplacement();
  void weightedDataSet();
};

#endif // OPENANN_TEST_DATA_SET_TEST_CASE_H_
