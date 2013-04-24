#pragma once

#include <Eigen/Dense>
#include <iostream>

namespace OpenANN {
namespace LibSVM {

int load(Eigen::MatrixXd& in, Eigen::MatrixXd& out, 
    const char* filename, int prefered_features = 0);

int load(Eigen::MatrixXd& in, Eigen::MatrixXd& out, 
    std::istream& stream, int prefered_features = 0);

void save(const Eigen::MatrixXd& in, const Eigen::MatrixXd& out, 
    const char* filename);

void save(const Eigen::MatrixXd& in, const Eigen::MatrixXd& out, 
    std::ostream& stream);

}} // namespace OpenANN::LibSVM
