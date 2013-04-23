#pragma once

#include <Eigen/Dense>
#include <vector>
#include <OpenANN/io/DataSet.h>

namespace OpenANN {

class Learner;

class DataSetView : public DataSet 
{
public:
  DataSetView(DataSet& dataset);

  DataSetView(const DataSetView& dataset);
  
  template<typename InputIt>
  DataSetView(DataSet& dataset, InputIt index_begin, InputIt index_end) 
  :  indices(index_begin, index_end), dataset(&dataset)
  {}
 
  virtual ~DataSetView() {}

  virtual int samples();

  virtual int inputs();

  virtual int outputs();

  virtual Eigen::VectorXd& getInstance(int i);

  virtual Eigen::VectorXd& getTarget(int i);

  virtual void finishIteration(Learner& learner);

  virtual DataSetView& shuffle();


private:
  // indices from the original dataset that are related to this subview.
  std::vector<int> indices;

  // reference to the original dataset interface
  DataSet* dataset;
};


void split(std::vector<DataSetView>& groups, DataSet& dataset, int number_of_groups);

void split(std::vector<DataSetView>& groups, DataSet& dataset, double ratio = 0.5);

DataSetView merge(std::vector<DataSetView>& groups);




} // namespace OpenANN
