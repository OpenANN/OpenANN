#pragma once

#include <OpenANN/Learner.h>
#include <OpenANN/io/DataSet.h>
#include <Eigen/Dense>

namespace OpenANN {

double sse(Learner& learner, DataSet& dataSet);
double mse(Learner& learner, DataSet& dataSet);
double rmse(Learner& learner, DataSet& dataSet);
double ce(Learner& learner, DataSet& dataSet);

double accuracy(Learner& learner, DataSet& dataSet);
Eigen::MatrixXd confusionMatrix(Learner& learner, DataSet& dataSet);

int oneOfCDecoding(const Eigen::VectorXd& target);

}
