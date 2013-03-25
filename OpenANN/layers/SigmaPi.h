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
        HigherOrderUnit(size_t space) {
            position.reserve(space);
        }

        std::vector<int> position;
        fpt* pWeight;
        fpt* pWeightDerivative;
    };

    struct HigherOrderNeuron 
    {
        std::vector<fpt> w;
        std::vector<fpt> wd;
        std::vector<HigherOrderUnit> units;
    };

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
    
    std::vector<HigherOrderNeuron> nodes;
    
 public:
  SigmaPi(OutputInfo info, bool bias, ActivationFunction act, fpt stdDev);

  virtual OutputInfo initialize(std::vector<fpt*>& parameterPointers,
          std::vector<fpt*>& parameterDerivativePointers);

  struct Constraint {
      virtual fpt operator() (int p1, int p2) const;
      virtual fpt operator() (int p1, int p2, int p3) const;
      virtual fpt operator() (int p1, int p2, int p3, int p4) const;
      virtual fpt operator() (const std::vector<int>& pos) const;
  };

  virtual SigmaPi& secondOrderNodes(int numbers, const Constraint* constrain = 0);

  virtual void initializeParameters();
  virtual void updatedParameters();
  virtual void forwardPropagate(Vt* x, Vt*& y);
  virtual void backpropagate(Vt* ein, Vt*& eout);
  virtual Vt& getOutput();
};



} // namespace OpenANN
