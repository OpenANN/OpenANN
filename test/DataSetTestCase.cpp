#include <OpenANN/util/AssertionMacros.h>
#include <OpenANN/io/DirectStorageDataSet.h>
#include <OpenANN/io/DataSetView.h>

#include "DataSetTestCase.h"

using namespace OpenANN;

DataSetTestCase::DataSetTestCase()
  : in(1, 5), out(1, 5)
{
  in << 1, 2, 3, 4, 5;
  out << 5, 4, 3, 2, 1;
}

void DataSetTestCase::run()
{
  RUN(DataSetTestCase, directStorageDataSets);
  RUN(DataSetTestCase, dataSetViews);
  RUN(DataSetTestCase, dataSetSplitsFromGroups);
  RUN(DataSetTestCase, dataSetSplitsFromRatio);
  RUN(DataSetTestCase, dataSetMerge);
}



void DataSetTestCase::directStorageDataSets()
{
  DirectStorageDataSet dataset(in, out);

  ASSERT_EQUALS(dataset.samples(), 5);
  ASSERT_EQUALS(dataset.inputs(), 1);
  ASSERT_EQUALS(dataset.outputs() ,1);

  ASSERT_EQUALS(dataset.getInstance(0).x(), 1);
  ASSERT_EQUALS(dataset.getInstance(1).x(), 2);
  ASSERT_EQUALS(dataset.getInstance(2).x(), 3);
  ASSERT_EQUALS(dataset.getInstance(3).x(), 4);
  ASSERT_EQUALS(dataset.getInstance(4).x(), 5);

  ASSERT_EQUALS(dataset.getTarget(0).x(), 5);
  ASSERT_EQUALS(dataset.getTarget(1).x(), 4);
  ASSERT_EQUALS(dataset.getTarget(2).x(), 3);
  ASSERT_EQUALS(dataset.getTarget(3).x(), 2);
  ASSERT_EQUALS(dataset.getTarget(4).x(), 1);
}



void DataSetTestCase::dataSetViews()
{
  DirectStorageDataSet dataset(in, out);
  DataSetView view(dataset);

  ASSERT_EQUALS(view.samples(), 0);
  ASSERT_EQUALS(view.inputs(), 1);
  ASSERT_EQUALS(view.outputs() ,1);
}



void DataSetTestCase::dataSetSplitsFromGroups()
{
  DirectStorageDataSet dataset(in, out);
  
  std::vector<int> X;
  std::vector<int> Y;
  std::vector<DataSetView> groups;

  split(groups, dataset, 3);

  ASSERT_EQUALS(groups.size(), 3);

  for(int i = 0; i < groups.size(); ++i) {
    for(int j = 0; j < groups.at(i).samples(); ++j) {
      X.push_back(groups.at(i).getInstance(j).x());
      Y.push_back(groups.at(i).getTarget(j).x());
    }
  }

  ASSERT_EQUALS(X.size(), in.size());
  ASSERT_EQUALS(Y.size(), out.size());

  OPENANN_CHECK(dataset.samples() == X.size() && dataset.samples() == Y.size());

  std::sort(X.begin(), X.end());
  std::sort(Y.begin(), Y.end());

  std::reverse(Y.begin(), Y.end());

  
  for(int i = 0; i < dataset.samples(); ++i) {
    ASSERT_EQUALS(X.at(i), dataset.getInstance(i).x());
    ASSERT_EQUALS(Y.at(i), dataset.getTarget(i).x());
  }
}


void DataSetTestCase::dataSetSplitsFromRatio()
{
  DirectStorageDataSet dataset(in, out);
  
  std::vector<int> X;
  std::vector<int> Y;
  std::vector<DataSetView> groups;

  split(groups, dataset, 0.01);

  ASSERT_EQUALS(groups.size(), 2);

  ASSERT(groups.front().samples() == 1);
  ASSERT(groups.back().samples() > 0);

  ASSERT(groups.front().samples() < groups.back().samples());

  for(int i = 0; i < groups.size(); ++i) {
    for(int j = 0; j < groups.at(i).samples(); ++j) {
      X.push_back(groups.at(i).getInstance(j).x());
      Y.push_back(groups.at(i).getTarget(j).x());
    }
  }

  ASSERT_EQUALS(X.size(), in.size());
  ASSERT_EQUALS(Y.size(), out.size());

  OPENANN_CHECK(dataset.samples() == X.size() && dataset.samples() == Y.size());

  std::sort(X.begin(), X.end());
  std::sort(Y.begin(), Y.end());

  std::reverse(Y.begin(), Y.end());
  
  for(int i = 0; i < dataset.samples(); ++i) {
    ASSERT_EQUALS(X.at(i), dataset.getInstance(i).x());
    ASSERT_EQUALS(Y.at(i), dataset.getTarget(i).x());
  }
}


void DataSetTestCase::dataSetMerge()
{
  DirectStorageDataSet dataset(in, out);
  DataSetView overall(dataset);
 
  std::vector<int> X;
  std::vector<int> Y;
  std::vector<DataSetView> groups;

  split(groups, dataset, 3);

  ASSERT_EQUALS(groups.size(), 3);

  merge(overall, groups);

  ASSERT_EQUALS(overall.samples(), dataset.samples());

  for (int i = 0; i < overall.samples(); ++i) {
    X.push_back(overall.getInstance(i).x());
    Y.push_back(overall.getTarget(i).x());
  }

  ASSERT_EQUALS(X.size(), in.size());
  ASSERT_EQUALS(Y.size(), out.size());

  OPENANN_CHECK(dataset.samples() == X.size() && dataset.samples() == Y.size());

  std::sort(X.begin(), X.end());
  std::sort(Y.begin(), Y.end());

  std::reverse(Y.begin(), Y.end());
  
  for(int i = 0; i < dataset.samples(); ++i) {
    ASSERT_EQUALS(X.at(i), dataset.getInstance(i).x());
    ASSERT_EQUALS(Y.at(i), dataset.getTarget(i).x());
  }
}


