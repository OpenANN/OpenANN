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
        for(int j = 0; j < nodes[i].size(); ++j) 
            w[nodes[i][j].weight] = rng.sampleNormalDistribution<fpt>() * stdDev;
}


void SigmaPi::updatedParameters()
{
}

void SigmaPi::forwardPropagate(Vt* x, Vt*& y, bool dropout)
{
    int J = nodes.size();
    this->x = x;

    for(int i = 0; i < nodes.size(); ++i) {
        HigherOrderNeuron& neuron = nodes[i];

        double sum = 0.0;

        for(int j = 0; j < neuron.size(); ++j) {
            HigherOrderUnit& unit = neuron[j];

            double korrelation = 1.0;

            for(int k = 0; k < unit.position.size(); ++k) {
               korrelation *= (*x)(unit.position.at(k));
            }

            sum += w[unit.weight] * korrelation;
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
        HigherOrderNeuron& neuron = nodes.at(i);

        double sum = 0.0;
        deltas(i) = (*error_in)(i) * yd(i);

        for(int j = 0; j < neuron.size(); ++j) {
            HigherOrderUnit& unit = neuron.at(j);

            double korrelation = 1.0;

            for(int k = 0; k < unit.position.size(); ++k) {
                korrelation *= (*x)(unit.position.at(k));
                e(unit.position.at(k)) += w[unit.weight] * deltas(i);
            }

            wd[unit.weight] = deltas(i) * korrelation;
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

        for(int p1 = 0; p1 < (I - 1); ++p1) {
            for(int p2 = p1 + 1; p2 < I; ++p2) {
                HigherOrderUnit snd_order_unit;

                snd_order_unit.position.push_back(p1);
                snd_order_unit.position.push_back(p2);

                if(constrain != 0) {
                    double ref = (*constrain)(p1, p2);
                    size_t found = neuron.size();

                    for(int j = 0; j < neuron.size(); ++j) {
                        if(std::fabs(w[neuron[j].weight] - ref) < 0.001) {
                            found = j;
                            j = neuron.size();
                        }
                    }

                    if(found >= neuron.size()) {
                        snd_order_unit.weight = w.size();
                        w.push_back(ref);
                        wd.push_back(ref);
                    } else { 
                        snd_order_unit.weight = neuron[found].weight;
                    }

                } else {
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


OutputInfo SigmaPi::initialize(std::vector<fpt*>& parameterPointers, std::vector<fpt*>& parameterDerivativePointers)
{
    int J = nodes.size();

    for(int i = 0; i < w.size(); ++i) {
        parameterPointers.push_back(&(w[i]));
        parameterDerivativePointers.push_back(&(wd[i]));
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

