#pragma once

#include <OpenANN/io/DirectStorageDataSet.h>
#include "Distorter.h"

class EnhancedDataSet : public OpenANN::DirectStorageDataSet
{
  int generateAfterIteration, iteration;
  Distorter& distorter;
  Eigen::MatrixXd* original;
public:
  EnhancedDataSet(Eigen::MatrixXd& in, Eigen::MatrixXd& out,
                  int generateAfterIteration, Distorter& distorter,
                  LogInfo logInfo = NONE,
                  OpenANN::Logger::Target target = OpenANN::Logger::CONSOLE)
    : DirectStorageDataSet(in, out, logInfo, target),
      generateAfterIteration(generateAfterIteration), iteration(0),
      distorter(distorter), original(&in)
  {
    DirectStorageDataSet::in = new Eigen::MatrixXd(original->rows(), original->cols());
    *DirectStorageDataSet::in = *original;
  }

  virtual ~EnhancedDataSet()
  {
    delete DirectStorageDataSet::in;
  }

  virtual void finishIteration(OpenANN::Learner& learner)
  {
    DirectStorageDataSet::finishIteration(learner);
    if(++iteration % generateAfterIteration == 0)
    {
      // Generate more data
      OPENANN_DEBUG << "Apply distortions...";
      distorter.createDistortionMap(28, 28); // TODO hardcoded...
      *DirectStorageDataSet::in = *original; // TODO do not copy
      distorter.applyDistortions(*DirectStorageDataSet::in);
    }
  }
};
