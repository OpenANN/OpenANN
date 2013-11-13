#ifndef OPENANN_IO_FANN_H_
#define OPENANN_IO_FANN_H_

#include <Eigen/Core>
#include <iostream>

namespace OpenANN
{

namespace FANN
{

/**
 * Read a FANN-encoded dataset from the filesystem and load
 * its values into given in- and output matrices.
 *
 * @param in input matrix with an unspecific dimension that
 *           will contain the data
 * @param out output matrix with an unspecific dimension that
 *            will contain the data.
 * @param filename name to the corresponding libsvm dataset file
 * @return the number of loaded instances from the dataset
 */
int load(Eigen::MatrixXd& in, Eigen::MatrixXd& out, const char* filename);

/**
 * Read a FANN-encoded dataset from any input stream and load
 * its values into given in- and output matrices.
 *
 * @param in input matrix with an unspecific dimension that
 *      will contain the data
 * @param out output matrix with an unspecific dimension that
 *      will contain the data.
 * @param stream general STL data stream for getting libsvm-encoded datasets
 * @return the number of loaded instances from the dataset
 */
int load(Eigen::MatrixXd& in, Eigen::MatrixXd& out, std::istream& stream);

/**
 * Export a given dataset represented by in- and output matrices into a libsvm file.
 *
 * @param in matrix containing all input values (features)
 * @param out matrix containt all output values (classes, targets)
 * @param filename name to the generating libsvm dataset file
 */
void save(const Eigen::MatrixXd& in, const Eigen::MatrixXd& out, const char* filename);

/**
 * Write a given dataset from in- and output matrices to an output stream in
 * libsvm format.
 *
 * @param in matrix containing all input values (features)
 * @param out matrix containt all output values (classes, targets)
 * @param stream general STL data stream that will receive libsvm encoded dataset
 */
void save(const Eigen::MatrixXd& in, const Eigen::MatrixXd& out, std::ostream& stream);

} // namespace FANN

} // namespace OpenANN

#endif // OPENANN_IO_FANN_H_
