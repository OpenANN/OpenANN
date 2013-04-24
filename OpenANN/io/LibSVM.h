#pragma once

#include <Eigen/Dense>
#include <iostream>

namespace OpenANN {
namespace LibSVM {

int load(Eigen::MatrixXd& in, Eigen::MatrixXd& out, const char* filename);
int load(Eigen::MatrixXd& in, Eigen::MatrixXd& out, std::istream& stream);

void save(const Eigen::MatrixXd& in, const Eigen::MatrixXd& out, const char* filename);
void save(const Eigen::MatrixXd& in, const Eigen::MatrixXd& out, std::ostream& stream);

}} // namespace OpenANN::LibSVM
