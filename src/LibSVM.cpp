#include <OpenANN/io/LibSVM.h>
#include <OpenANN/util/AssertionMacros.h>
#include <fstream>
#include <string>
#include <set>
#include <vector>
#include <algorithm>

namespace OpenANN {
namespace LibSVM {

int load(Eigen::MatrixXd& in, Eigen::MatrixXd& out, const char* filename, int cols)
{
  std::ifstream fin(filename, std::ios_base::in);
  int count = load(in, out, fin, cols);
  fin.close();

  return count;
}


int load(Eigen::MatrixXd& in, Eigen::MatrixXd& out, std::istream& stream, int cols)
{
  std::vector<std::string> features;
  std::vector<int> targets;
  std::set<int> classes;
  int minimal_features = cols;
  int instances = 0;

  // preprocessing for retrieving all necessary information
  while(stream.good()) {
    std::vector<std::string> line;
    std::string buffer, token;
    std::getline(stream, buffer);
    
    std::stringstream ss(buffer);
    while(ss >> token) {
      if(token.size() > 0)
        line.push_back(token);
    }

    if(line.size() > 0) {
      std::string& classStr = line.front();
      std::string& last = line.back();

      int y = std::atoi(classStr.c_str());
      classes.insert(y);
      targets.push_back(y);

      minimal_features = 
        std::max(
            minimal_features, 
            std::atoi(last.substr(0, last.find_first_of(':')).c_str())
        );

      std::copy(line.begin() + 1, line.end(), std::back_inserter(features));
      features.push_back("");
      instances++;
    }
  }

  // prepare matrices
  in.resize(minimal_features, instances);
  
  if(classes.size() > 2)
    out.resize(classes.size(), instances);
  else
    out.resize(1, instances);
  
  in.setZero();
  out.setZero();

  // setup output matrix from collected targets
  for(int i = 0; i < targets.size(); ++i) {
    if(classes.size() > 2) {
      out(targets[i] - 1, i) = 1.0;
    } else {
      out(0, i) = targets.at(i);
    }
  }

  // setup input matrix from collected tokens
  for(int i = 0, column = 0; i < features.size(); ++i) {
    std::string& tok = features.at(i);

    if(!tok.empty()) {
      size_t pos = tok.find_first_of(':');
      int index = std::atoi(tok.substr(0, pos).c_str());
      double value = std::atof(tok.substr(pos + 1).c_str());

      // index counting is different between supervised- and unsupervised 
      // datasets
      if(classes.size() > 1) 
        index--;

      in(index, column) = value;
    } else {
      column++;
    }
  }

  return instances;
}


void save(const Eigen::MatrixXd& in, const Eigen::MatrixXd& out, const char* filename)
{
  std::ofstream fout(filename, std::ios_base::out);
  save(in, out, fout);
  fout.close();
}


void save(const Eigen::MatrixXd& in, const Eigen::MatrixXd& out, std::ostream& stream)
{
  OPENANN_CHECK_EQUALS(in.cols(), out.cols());

  for(int i = 0; i < in.cols(); ++i) {
    if(out.rows() > 1)
      stream << static_cast<int>(out.col(i).maxCoeff());
    else
      stream << out(0, i);

    for(int j = 0; j < in.rows(); ++j) {
      if(std::fabs(in(j, i)) > 0.0e-20)
        stream << " " << j + 1 << ":" << in(j, i);
    }

    stream << std::endl;
  }
}



}} // namespace OpenANN::LibSVM
