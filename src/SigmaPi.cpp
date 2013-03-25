#include <layers/SigmaPi.h>
#include <Random.h>
#include <stdexcept>

namespace OpenANN {

fpt SigmaPi::Constraint::operator() (int p1, int p2) const 
{
    throw std::runtime_error("constrain operator (p1, p2) must be implemented");
    return 0.0;
}

fpt SigmaPi::Constraint::operator() (int p1, int p2, int p3) const 
{
    throw std::runtime_error("constrain operator (p1, p2, p3) must be implemented");
    return 0.0;
}

fpt SigmaPi::Constraint::operator() (int p1, int p2, int p3, int p4) const 
{
    throw std::runtime_error("constrain operator (p1, p2, p3, p4) must be implemented");
    return 0.0;
}

fpt SigmaPi::Constraint::operator() (const std::vector<int>& pos) const 
{
    throw std::runtime_error("constrain operator (pos, ...) must be implemented");
    return 0.0;
}


SigmaPi::SigmaPi(OutputInfo info, bool bias, ActivationFunction act, fpt stdDev)
    : info(info), bias(bias), act(act), stdDev(stdDev), x(0), e(info.outputs())
{
}


void SigmaPi::initializeParameters()
{
    RandomNumberGenerator rng;
    for(int i = 0; i < nodes.size(); ++i)
        for(int j = 0; j < nodes[i].units.size(); ++j) 
            (*nodes[i].units[j].pWeight) = rng.sampleNormalDistribution<fpt>() * stdDev;
}


void SigmaPi::updatedParameters()
{
}

void SigmaPi::forwardPropagate(Vt* x, Vt*& y)
{
    int J = nodes.size();
    this->x = x;

    for(int i = 0; i < nodes.size(); ++i) {
        HigherOrderNeuron& neuron = nodes[i];

        double sum = 0.0;

        for(int j = 0; j < neuron.units.size(); ++j) {
            HigherOrderUnit& unit = neuron.units[j];

            double korrelation = 1.0;

            for(int k = 0; k < unit.position.size(); ++k) {
                korrelation *= (*x)(unit.position.at(k));
            }

            sum += (*unit.pWeight) * korrelation;
        }

        a(i) = sum;
    }

    activationFunction(act, a, this->y);

    if(bias)
        this->y(J) = 1.0;

    y = &(this->y);
}


void SigmaPi::backpropagate(Vt* error_in, Vt*& error_out)
{
    e.fill(0.0);

    activationFunctionDerivative(act, y, yd);

    for(int i = 0; i < nodes.size(); ++i) {
        HigherOrderNeuron& neuron = nodes[i];

        double sum = 0.0;
        deltas(i) = (*error_in)(i) * yd(i);

        for(int j = 0; j < neuron.units.size(); ++j) {
            HigherOrderUnit& unit = neuron.units[j];

            double korrelation = 1.0;

            for(int k = 0; k < unit.position.size(); ++k) {
                korrelation *= (*x)(unit.position.at(k));
                e(unit.position.at(k)) += (*unit.pWeight) * deltas(i);
            }

            (*unit.pWeightDerivative) = deltas(i) * korrelation;
        }
    }

    error_out = &e;
}


Vt& SigmaPi::getOutput()
{
    return y;
}

SigmaPi& SigmaPi::secondOrderNodes(int numbers, const Constraint* constrain)
{
    int I = info.outputs();

    for(int i = 0; i < numbers; ++i) {
        HigherOrderNeuron neuron;

        for(int p1 = 0; p1 < I - 1; ++p1) {
            for(int p2 = p1 + 1; p2 < I; ++p2) {
                HigherOrderUnit snd_order_unit(2);

                snd_order_unit.position.push_back(p1);
                snd_order_unit.position.push_back(p2);

                if(constrain != 0) {
                   double weight = (*constrain)(p1, p2);

                   std::vector<fpt>::iterator dt;

                   dt = find(neuron.w.begin(), neuron.w.end(), weight);

                   if(dt == neuron.w.end()) {
                       neuron.w.push_back(weight);
                       neuron.wd.push_back(weight);

                       snd_order_unit.pWeight = &neuron.w.back();
                       snd_order_unit.pWeightDerivative = &neuron.wd.back();
                   } else {
                       int index = (dt - neuron.w.begin());
                       
                       snd_order_unit.pWeight = &neuron.w[index];
                       snd_order_unit.pWeightDerivative = &neuron.wd[index];
                   }
                } else {
                    neuron.w.push_back(0.0);
                    neuron.wd.push_back(0.0);

                    snd_order_unit.pWeight = &neuron.w.back();
                    snd_order_unit.pWeightDerivative = &neuron.wd.back();
                }

                neuron.units.push_back(snd_order_unit);                
            }
        }

        nodes.push_back(neuron);
    }
}


OutputInfo SigmaPi::initialize(std::vector<fpt*>& parameterPointers, std::vector<fpt*>& parameterDerivativePointers)
{
    int J = nodes.size();

    for(int i = 0; i < J; ++i) {
        for(int j = 0; j < nodes[i].w.size(); ++j) {
            parameterPointers.push_back(&(nodes[i].w[j]));
            parameterDerivativePointers.push_back(&(nodes[i].wd[j]));
        }
    }

    y.resize(J + bias);
    yd.resize(J);
    deltas.resize(J);
    a.resize(J);

    if(bias)
        y(J) = 1.0;

    initializeParameters();

    OutputInfo info;
    info.bias = bias;
    info.dimensions.push_back(J);
    return info;
}


}

