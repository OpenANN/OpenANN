#include <OpenANN/io/DataSetView.h>
#include <OpenANN/util/AssertionMacros.h>
#include <OpenANN/Learner.h>
#include <algorithm>
#include <ctime>
#include <cmath>
#include <cstdlib>

namespace OpenANN
{

DataSetView::DataSetView(const DataSetView& ds)
  : indices(ds.indices), dataset(ds.dataset)
{
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
  std::random_shuffle(indices.begin(), indices.end());

  return *this;
}



void split(std::vector<DataSetView>& groups, DataSet& dataset,
           int numberOfGroups, bool shuffling)
{
  OPENANN_CHECK(numberOfGroups > 1);
  std::vector<int> indices;

  indices.reserve(dataset.samples());
  groups.reserve(numberOfGroups);

  for(int i = 0; i < dataset.samples(); ++i)
    indices.push_back(i);

  int samplesPerGroup = std::floor(dataset.samples() / numberOfGroups + 0.5);

  if(shuffling)
    std::random_shuffle(indices.begin(), indices.end());

  for(int i = 0; i < numberOfGroups; ++i)
  {
    std::vector<int>::iterator it = indices.begin() + i * samplesPerGroup;

    if(i < numberOfGroups - 1)
      groups.push_back(DataSetView(dataset, it, it + samplesPerGroup));
    else
      groups.push_back(DataSetView(dataset, it, indices.end()));
  }
}



void split(std::vector<DataSetView>& groups, DataSet& dataset, double ratio,
           bool shuffling)
{
  OPENANN_CHECK_WITHIN(ratio, 0.0, 1.0);
  std::vector<int> indices;

  indices.reserve(dataset.samples());
  groups.reserve(2);

  for(int i = 0; i < dataset.samples(); ++i)
    indices.push_back(i);

  int samples = std::ceil(ratio * dataset.samples());

  if(shuffling)
    std::random_shuffle(indices.begin(), indices.end());

  groups.push_back(DataSetView(dataset, indices.begin(), indices.begin() + samples));
  groups.push_back(DataSetView(dataset, indices.begin() + samples, indices.end()));
}



void merge(DataSetView& merging, std::vector<DataSetView>& groups)
{
  OPENANN_CHECK(!groups.empty());

  for(int i = 0; i < groups.size(); ++i)
  {
    OPENANN_CHECK(merging.dataset == groups.at(i).dataset);

    std::copy(groups.at(i).indices.begin(), groups.at(i).indices.end(),
              std::back_inserter(merging.indices));
  }
}

} // namespace OpenANN
