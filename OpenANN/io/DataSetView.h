#pragma once

#include <Eigen/Dense>
#include <vector>
#include <OpenANN/io/DataSet.h>

namespace OpenANN {

class Learner;

class DataSetView : public DataSet 
{
public:
  DataSetView(const DataSetView& dataset);
  
  DataSetView(DataSet& dataset) : dataset(&dataset) 
  {}
  
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

  // friend declaration for direct indices access
  friend void merge(DataSetView& merging, std::vector<DataSetView>& groups);
};


void split(std::vector<DataSetView>& groups, DataSet& dataset, int number_of_groups, bool shuffling = true);

void split(std::vector<DataSetView>& groups, DataSet& dataset, double ratio = 0.5, bool shuffling = true);

void merge(DataSetView& merging, std::vector<DataSetView>& groups);




} // namespace OpenANN
