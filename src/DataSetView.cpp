#include <OpenANN/io/DataSetView.h>
#include <OpenANN/util/AssertionMacros.h>
#include <OpenANN/Learner.h>
#include <algorithm>
#include <ctime>
#include <cmath>
#include <cstdlib>

namespace OpenANN {

int default_random(int i)
{
  static bool initialized = false;

  if(!initialized) {
    std::srand(unsigned(std::time(0)));
    initialized = true;
  }

  return std::rand() % i;
}

  DataSetView::DataSetView(const DataSetView& ds)
  : indices(ds.indices), dataset(ds.dataset)
{
}

DataSetView::DataSetView(DataSet& dataset) : dataset(&dataset)
{
  for(int i = 0; i < dataset.samples(); ++i)
    indices.push_back(i);
}


int DataSetView::samples() 
{
  return indices.size();
}


int DataSetView::inputs()
{
  return dataset->inputs();
}


int DataSetView::outputs()
{
  return dataset->outputs();
}


Eigen::VectorXd& DataSetView::getInstance(int i)
{
  OPENANN_CHECK_WITHIN(i, 0, samples() - 1);
  return dataset->getInstance(indices.at(i));
}


Eigen::VectorXd& DataSetView::getTarget(int i)
{
  OPENANN_CHECK_WITHIN(i, 0, samples() - 1);
  return dataset->getTarget(indices.at(i));
}


void DataSetView::finishIteration(Learner& learner)
{
  dataset->finishIteration(learner);
}


DataSetView& DataSetView::shuffle()
{
  std::random_shuffle(indices.begin(), indices.end(), default_random);

  return *this;
}


void split(std::vector<DataSetView>& groups, DataSet& dataset, int number_of_groups)
{
  OPENANN_CHECK(number_of_groups > 1);

  int samples_per_group = dataset.samples() / number_of_groups;

  std::vector<int> indices;

  for(int i = 0; i < dataset.samples(); ++i)
    indices.push_back(i);

  std::random_shuffle(indices.begin(), indices.end(), default_random);
  std::vector<int>::iterator it;

  for(it = indices.begin(); it != indices.end(); it += samples_per_group)
    groups.push_back(DataSetView(dataset, it, it + samples_per_group));
}

void split(std::vector<DataSetView>& groups, DataSet& dataset, double ratio)
{

}

DataSetView merge(std::vector<DataSetView>& groups)
{
  DataSetView& first = groups.front();

  return first;
}




}
