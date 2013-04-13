#include <OpenANN/layers/SigmaPi.h>
#include <OpenANN/util/Random.h>
#include <OpenANN/util/OpenANNException.h>

namespace OpenANN {

fpt SigmaPi::Constraint::operator() (int p1, int p2) const 
{
    throw OpenANNException("Constrain operator (p1, p2) must be implemented");
    return 0.0;
}

fpt SigmaPi::Constraint::operator() (int p1, int p2, int p3) const 
{
    throw OpenANNException("Constrain operator (p1, p2, p3) must be implemented");
    return 0.0;
}

fpt SigmaPi::Constraint::operator() (int p1, int p2, int p3, int p4) const 
{
    throw OpenANNException("Constrain operator (p1, p2, p3, p4) must be implemented");
    return 0.0;
}


bool SigmaPi::Constraint::isDefault() const {
    return false;
}


struct NoConstraint : public OpenANN::SigmaPi::Constraint
{
    virtual bool isDefault() const {
        return true;
    }
};



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

    for(int i = 0; i < wd.size(); ++i)
        wd[i] = 0.0;

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

            wd[unit.weight] += deltas(i) * korrelation;
        }
    }

    error_out = &e;
}


Vt& SigmaPi::getOutput()
{
    return y;
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
    int I = info.outputs();

    for(int i = 0; i < numbers; ++i) {
        HigherOrderNeuron neuron;

        for(int p1 = 0; p1 < (I - 1); ++p1) {
            for(int p2 = p1 + 1; p2 < I; ++p2) {
                HigherOrderUnit snd_order_unit;

                snd_order_unit.position.push_back(p1);
                snd_order_unit.position.push_back(p2);

                if(!constraint.isDefault()) {
                    double ref = constraint(p1, p2);
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

SigmaPi& SigmaPi::thirdOrderNodes(int numbers, const Constraint& constraint)
{
    int I = info.outputs();

    for(int i = 0; i < numbers; ++i) {
        HigherOrderNeuron neuron;

        for(int p1 = 0; p1 < (I - 2); ++p1) {
            for(int p2 = p1 + 1; p2 < (I - 1); ++p2) {
                for(int p3 = p2 + 1; p3 < I; ++p3) {
                    HigherOrderUnit honn_unit;

                    honn_unit.position.push_back(p1);
                    honn_unit.position.push_back(p2);
                    honn_unit.position.push_back(p3);

                    if(!constraint.isDefault()) {
                        double ref = constraint(p1, p2, p3);
                        size_t found = neuron.size();

                        for(int j = 0; j < neuron.size(); ++j) {
                            if(std::fabs(w[neuron[j].weight] - ref) < 0.001) {
                                found = j;
                                j = neuron.size();
                            }
                        }

                        if(found >= neuron.size()) {
                            honn_unit.weight = w.size();
                            w.push_back(ref);
                            wd.push_back(ref);
                        } else { 
                            honn_unit.weight = neuron[found].weight;
                        }
                    } else {
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
    int I = info.outputs();

    for(int i = 0; i < numbers; ++i) {
        HigherOrderNeuron neuron;

        for(int p1 = 0; p1 < (I - 3); ++p1) {
            for(int p2 = p1 + 1; p2 < (I - 2); ++p2) {
                for(int p3 = p2 + 1; p3 < (I - 1); ++p3) {
                    for(int p4 = p3 + 1; p4 < I; ++p4) {
                        HigherOrderUnit honn_unit;

                        honn_unit.position.push_back(p1);
                        honn_unit.position.push_back(p2);
                        honn_unit.position.push_back(p3);
                        honn_unit.position.push_back(p4);

                        if(!constraint.isDefault()) {
                            double ref = constraint(p1, p2, p3, p4);
                            size_t found = neuron.size();

                            for(int j = 0; j < neuron.size(); ++j) {
                                if(std::fabs(w[neuron[j].weight] - ref) < 0.001) {
                                    found = j;
                                    j = neuron.size();
                                }
                            }

                            if(found >= neuron.size()) {
                                honn_unit.weight = w.size();
                                w.push_back(ref);
                                wd.push_back(ref);
                            } else { 
                                honn_unit.weight = neuron[found].weight;
                            }
                        } else {
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



}

