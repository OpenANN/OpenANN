#pragma once

#include <Eigen/Dense>

namespace OpenANN {

class Optimizer;
class Learner;
class DataSet;

double sse(Learner& learner, DataSet& dataSet);
double mse(Learner& learner, DataSet& dataSet);
double rmse(Learner& learner, DataSet& dataSet);
double ce(Learner& learner, DataSet& dataSet);

double accuracy(Learner& learner, DataSet& dataSet);
Eigen::MatrixXd confusionMatrix(Learner& learner, DataSet& dataSet);

void crossValidation(int folds, Learner& learner, DataSet& dataSet, Optimizer& opt);

int oneOfCDecoding(const Eigen::VectorXd& target);

}
