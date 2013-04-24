#pragma once

#include <OpenANN/Learner.h>
#include <OpenANN/io/DataSet.h>
#include <Eigen/Dense>

namespace OpenANN {

double sse(const Learner& learner, const DataSet& dataSet);
double mse(const Learner& learner, const DataSet& dataSet);
double rmse(const Learner& learner, const DataSet& dataSet);
double ce(const Learner& learner, const DataSet& dataSet);

double accuracy(const Learner& learner, const DataSet& dataSet);
Eigen::MatrixXd confusionMatrix(const Learner& learner, const DataSet& dataSet);

int oneOfCDecoding(const Eigen::VectorXd& target);

}
