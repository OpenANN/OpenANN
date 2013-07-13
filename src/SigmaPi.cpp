#include <OpenANN/layers/SigmaPi.h>
#include <OpenANN/util/Random.h>
#include <OpenANN/util/OpenANNException.h>

namespace OpenANN
{

double SigmaPi::Constraint::operator()(int p1, int p2) const
{
  throw OpenANNException("Constrain operator (p1, p2) must be implemented");
  return 0.0;
}

double SigmaPi::Constraint::operator()(int p1, int p2, int p3) const
{
  throw OpenANNException("Constrain operator (p1, p2, p3) must be implemented");
  return 0.0;
}

double SigmaPi::Constraint::operator()(int p1, int p2, int p3, int p4) const
{
  throw OpenANNException("Constrain operator (p1, p2, p3, p4) must be implemented");
  return 0.0;
}


bool SigmaPi::Constraint::isDefault() const
{
  return false;
}


struct NoConstraint : public OpenANN::SigmaPi::Constraint
{
  virtual bool isDefault() const
  {
    return true;
  }
};



SigmaPi::SigmaPi(OutputInfo info, bool bias, ActivationFunction act, double stdDev)
  : info(info), bias(bias), act(act), stdDev(stdDev),
    x(1, info.outputs() + bias), e(1, info.outputs())
{
  if(bias)
    x(info.outputs()) = 1.0;
}


void SigmaPi::initializeParameters()
{
  RandomNumberGenerator rng;
  for(int i = 0; i < nodes.size(); ++i)
    for(int j = 0; j < nodes[i].size(); ++j)
      w[nodes[i][j].weight] = rng.sampleNormalDistribution<double>() * stdDev;
}


void SigmaPi::updatedParameters()
{
}

void SigmaPi::forwardPropagate(Eigen::MatrixXd* x, Eigen::MatrixXd*& y,
                               bool dropout)
{
  const int N = x->rows();
  this->x.conservativeResize(N, Eigen::NoChange);
  a.conservativeResize(N, Eigen::NoChange);
  this->x.rightCols(1).setZero();
  this->y.conservativeResize(N, Eigen::NoChange);
  this->x.leftCols(info.outputs()) = *x;

  #pragma omp parallel for
  for(int instance = 0; instance < N; instance++)
  {
    int i = 0;
    for(HigherOrderNeuron* n = &nodes.front(); n <= &nodes.back(); ++n)
    {
      double sum = 0.0;

      for(HigherOrderUnit* u = &n->front(); u <= &n->back(); ++u)
      {

        double korrelation = 1.0;

        for(int k = 0; k < u->position.size(); ++k)
        {
          korrelation *= (*x)(instance, u->position.at(k));
        }

        sum = sum + w[u->weight] * korrelation;
      }

      a(instance, i++) = sum;
    }
  }

  activationFunction(act, a, this->y);

  y = &(this->y);
}


void SigmaPi::backpropagate(Eigen::MatrixXd* error_in,
                            Eigen::MatrixXd*& error_out,
                            bool backpropToPrevious)
{
  const int N = a.rows();
  e.conservativeResize(N, Eigen::NoChange);
  deltas.conservativeResize(N, Eigen::NoChange);
  yd.conservativeResize(N, Eigen::NoChange);
  e.setZero();

  for(int i = 0; i < wd.size(); ++i)
    wd[i] = 0.0;

  activationFunctionDerivative(act, y, yd);

  for(int instance = 0; instance < N; instance++)
  {
    int i = 0;
    for(HigherOrderNeuron* n = &nodes.front(); n <= &nodes.back(); ++n)
    {
      deltas(instance, i) = (*error_in)(instance, i) * yd(instance, i);

      for(HigherOrderUnit* u = &n->front(); u <= &n->back(); ++u)
      {
        double korrelation = 1.0;

        for(int k = 0; k < u->position.size(); ++k)
        {
          int index = u->position.at(k);
          korrelation *= x(instance, index);
          e(instance, index) += w[u->weight] * deltas(instance, i);
        }

        wd[u->weight] += deltas(instance, i) * korrelation;
      }

      ++i;
    }
  }

  error_out = &e;
}


Eigen::MatrixXd& SigmaPi::getOutput()
{
  return y;
}

OutputInfo SigmaPi::initialize(std::vector<double*>& parameterPointers, std::vector<double*>& parameterDerivativePointers)
{
  int J = nodes.size();

  for(int i = 0; i < w.size(); ++i)
  {
    parameterPointers.push_back(&(w[i]));
    parameterDerivativePointers.push_back(&(wd[i]));
  }

  y.resize(1, J + bias);
  yd.resize(1, J);
  deltas.resize(1, J);
  a.resize(1, J);

  initializeParameters();

  OutputInfo info;
  info.dimensions.push_back(J);
  return info;
}


SigmaPi& SigmaPi::secondOrderNodes(int numbers)
{
  NoConstraint constrain;
  return secondOrderNodes(numbers, constrain);
}

SigmaPi& SigmaPi::thirdOrderNodes(int numbers)
{
  NoConstraint constrain;
  return thirdOrderNodes(numbers, constrain);
}

SigmaPi& SigmaPi::fourthOrderNodes(int numbers)
{
  NoConstraint constrain;
  return fourthOrderNodes(numbers, constrain);
}



SigmaPi& SigmaPi::secondOrderNodes(int numbers, const Constraint& constraint)
{
  int I = info.outputs() + bias;

  for(int i = 0; i < numbers; ++i)
  {
    HigherOrderNeuron neuron;

    for(int p1 = 0; p1 < (I - 1); ++p1)
    {
      for(int p2 = p1 + 1; p2 < I; ++p2)
      {
        HigherOrderUnit snd_order_unit;

        snd_order_unit.position.push_back(p1);
        snd_order_unit.position.push_back(p2);

        if(!constraint.isDefault())
        {
          double ref = constraint(p1, p2);
          size_t found = neuron.size();

          for(int j = 0; j < neuron.size(); ++j)
          {
            if(std::fabs(w[neuron[j].weight] - ref) < 0.001)
            {
              found = j;
              j = neuron.size();
            }
          }

          if(found >= neuron.size())
          {
            snd_order_unit.weight = w.size();
            w.push_back(ref);
            wd.push_back(ref);
          }
          else
          {
            snd_order_unit.weight = neuron[found].weight;
          }
        }
        else
        {
          snd_order_unit.weight = w.size();

          w.push_back(0.0);
          wd.push_back(0.0);
        }

        neuron.push_back(snd_order_unit);
      }
    }

    nodes.push_back(neuron);
  }

  return *this;
}

SigmaPi& SigmaPi::thirdOrderNodes(int numbers, const Constraint& constraint)
{
  int I = info.outputs() + bias;

  for(int i = 0; i < numbers; ++i)
  {
    HigherOrderNeuron neuron;

    for(int p1 = 0; p1 < (I - 2); ++p1)
    {
      for(int p2 = p1 + 1; p2 < (I - 1); ++p2)
      {
        for(int p3 = p2 + 1; p3 < I; ++p3)
        {
          HigherOrderUnit honn_unit;

          honn_unit.position.push_back(p1);
          honn_unit.position.push_back(p2);
          honn_unit.position.push_back(p3);

          if(!constraint.isDefault())
          {
            double ref = constraint(p1, p2, p3);
            size_t found = neuron.size();

            for(int j = 0; j < neuron.size(); ++j)
            {
              if(std::fabs(w[neuron[j].weight] - ref) < 0.001)
              {
                found = j;
                j = neuron.size();
              }
            }

            if(found >= neuron.size())
            {
              honn_unit.weight = w.size();
              w.push_back(ref);
              wd.push_back(ref);
            }
            else
            {
              honn_unit.weight = neuron[found].weight;
            }
          }
          else
          {
            honn_unit.weight = w.size();

            w.push_back(0.0);
            wd.push_back(0.0);
          }

          neuron.push_back(honn_unit);
        }
      }
    }

    nodes.push_back(neuron);
  }

  return *this;
}


SigmaPi& SigmaPi::fourthOrderNodes(int numbers, const Constraint& constraint)
{
  int I = info.outputs() + bias;

  for(int i = 0; i < numbers; ++i)
  {
    HigherOrderNeuron neuron;

    for(int p1 = 0; p1 < (I - 3); ++p1)
    {
      for(int p2 = p1 + 1; p2 < (I - 2); ++p2)
      {
        for(int p3 = p2 + 1; p3 < (I - 1); ++p3)
        {
          for(int p4 = p3 + 1; p4 < I; ++p4)
          {
            HigherOrderUnit honn_unit;

            honn_unit.position.push_back(p1);
            honn_unit.position.push_back(p2);
            honn_unit.position.push_back(p3);
            honn_unit.position.push_back(p4);

            if(!constraint.isDefault())
            {
              double ref = constraint(p1, p2, p3, p4);
              size_t found = neuron.size();

              for(int j = 0; j < neuron.size(); ++j)
              {
                if(std::fabs(w[neuron[j].weight] - ref) < 0.001)
                {
                  found = j;
                  j = neuron.size();
                }
              }

              if(found >= neuron.size())
              {
                honn_unit.weight = w.size();
                w.push_back(ref);
                wd.push_back(ref);
              }
              else
              {
                honn_unit.weight = neuron[found].weight;
              }
            }
            else
            {
              honn_unit.weight = w.size();

              w.push_back(0.0);
              wd.push_back(0.0);
            }

            neuron.push_back(honn_unit);
          }
        }
      }
    }

    nodes.push_back(neuron);
  }

  return *this;
}

Eigen::VectorXd SigmaPi::getParameters()
{
  throw OpenANNException("SigmaPi::getParameters() is not implemented!");
}

}

