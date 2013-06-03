#include <OpenANN/io/FANN.h>
#include <OpenANN/util/AssertionMacros.h>
#include <fstream>

namespace OpenANN
{

namespace FANN
{

int load(Eigen::MatrixXd& in, Eigen::MatrixXd& out, const char* filename)
{
  std::ifstream fin(filename, std::ios_base::in);
  int count = load(in, out, fin);
  fin.close();

  return count;
}


int load(Eigen::MatrixXd& in, Eigen::MatrixXd& out, std::istream& stream)
{
  int D, F, N;
  stream >> N >> D >> F;

  in.resize(N, D);
  out.resize(N, F);

  for(int n = 0; n < N; n++)
  {
    for(int d = 0; d < D; d++)
      stream >> in(n, d);
    for(int f = 0; f < F; f++)
      stream >> out(n, f);
  }

  return N;
}


void save(const Eigen::MatrixXd& in, const Eigen::MatrixXd& out, const char* filename)
{
  std::ofstream fout(filename, std::ios_base::out);
  save(in, out, fout);
  fout.close();
}


void save(const Eigen::MatrixXd& in, const Eigen::MatrixXd& out, std::ostream& stream)
{
  OPENANN_CHECK_EQUALS(in.rows(), out.rows());

  int N = in.rows();
  int D = in.cols();
  int F = out.cols();

  stream << N << " " << D << " " << F << std::endl;
  for(int n = 0; n < N; n++)
  {
    stream << in.row(n) << std::endl;
    stream << out.row(n) << std::endl;
  }
}

}

}
