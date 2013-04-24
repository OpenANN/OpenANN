#include <OpenANN/io/LibSvm.h>
#include <fstream>

namespace OpenANN {
namespace LibSVM {

int load(Eigen::MatrixXd& in, Eigen::MatrixXd& out, const char* filename)
{
  std::ifstream fin(filename, std::ios_base::in);
  int count = load(in, out, fin);
  fin.close();

  return count;
}


int load(Eigen::MatrixXd& in, Eigen::MatrixXd& out, std::istream& stream)
{
}


void save(const Eigen::MatrixXd& in, const Eigen::MatrixXd& out, const char* filename)
{
  std::ofstream fout(filename, std::ios_base::out);
  save(in, out, fout);
  fout.close();
}


void save(const Eigen::MatrixXd& in, const Eigen::MatrixXd& out, std::ostream& stream)
{
}



}} // namespace OpenANN::LibSVM
