#pragma once

#include <Eigen/Dense>
#include <vector>
#include <layers/Layer.h>
#include <ActivationFunctions.h>

namespace OpenANN {

class SigmaPi : public Layer 
{
 protected:
    struct HigherOrderUnit 
    {
        std::vector<int> position;
        size_t weight;
    };

    typedef std::vector<HigherOrderUnit> HigherOrderNeuron;

    OutputInfo info;
    bool bias;
    ActivationFunction act;
    fpt stdDev;

    Vt* x;
    Vt a;
    Vt y;
    Vt yd;
    Vt deltas;
    Vt e;
 
    std::vector<fpt> w;
    std::vector<fpt> wd;
    std::vector<HigherOrderNeuron> nodes;
    
 public:
  SigmaPi(OutputInfo info, bool bias, ActivationFunction act, fpt stdDev);

  virtual OutputInfo initialize(std::vector<fpt*>& parameterPointers,
          std::vector<fpt*>& parameterDerivativePointers);

  struct Constraint {
      virtual fpt operator() (int p1, int p2) const;
      virtual fpt operator() (int p1, int p2, int p3) const;
      virtual fpt operator() (int p1, int p2, int p3, int p4) const;
      virtual bool isDefault() const;
  };

  virtual SigmaPi& secondOrderNodes(int numbers, const Constraint& constrain);
  virtual SigmaPi& secondOrderNodes(int numbers);
  
  virtual SigmaPi& thirdOrderNodes(int numbers, const Constraint& constrain);
  virtual SigmaPi& thirdOrderNodes(int numbers);

  virtual SigmaPi& fourthOrderNodes(int numbers, const Constraint& constrain);
  virtual SigmaPi& fourthOrderNodes(int numbers);

  virtual size_t nodenumber() const { return nodes.size(); };
  virtual size_t parameter() const { return w.size(); };
  
  virtual void initializeParameters();
  virtual void updatedParameters();
  virtual void forwardPropagate(Vt* x, Vt*& y, bool dropout = false);
  virtual void backpropagate(Vt* ein, Vt*& eout);
  virtual Vt& getOutput();
};



} // namespace OpenANN
