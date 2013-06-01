#ifndef ENHANCED_DATA_SET_H_
#define ENHANCED_DATA_SET_H_

#include <OpenANN/io/DirectStorageDataSet.h>
#include "Distorter.h"

class EnhancedDataSet : public OpenANN::DirectStorageDataSet
{
  int generateAfterIteration, iteration;
  Distorter& distorter;
  Eigen::MatrixXd* original;
  int pixelsPerDim;
public:
  EnhancedDataSet(Eigen::MatrixXd& in, Eigen::MatrixXd& out,
                  int generateAfterIteration, Distorter& distorter)
    : DirectStorageDataSet(&in, &out),
      generateAfterIteration(generateAfterIteration), iteration(0),
      distorter(distorter), original(&in), pixelsPerDim(std::sqrt(in.rows()))
  {
    DirectStorageDataSet::in = new Eigen::MatrixXd(original->rows(),
                                                   original->cols());
    *DirectStorageDataSet::in = *original;
  }

  virtual ~EnhancedDataSet()
  {
    delete DirectStorageDataSet::in;
  }

  virtual void finishIteration(OpenANN::Learner& learner)
  {
    DirectStorageDataSet::finishIteration(learner);
    if(iteration++ % generateAfterIteration == 0)
      distort(); // Generate more data
  }

  void distort()
  {
    OPENANN_INFO << "Apply distortions...";
    *DirectStorageDataSet::in = *original; // TODO do not copy
    distorter.applyDistortions(*DirectStorageDataSet::in, pixelsPerDim,
                               pixelsPerDim);
    OPENANN_INFO << "Done.";
  }
};

#endif // ENHANCED_DATA_SET_H_
