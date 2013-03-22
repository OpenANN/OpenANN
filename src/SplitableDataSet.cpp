#include <io/SplitableDataSet.h>
#include <AssertionMacros.h>
#include <Learner.h>
#include <algorithm>
#include <ctime>
#include <cmath>
#include <cstdlib>

namespace OpenANN {

int default_random(int i) 
{
    return std::rand() % i; 
}

SplitableDataSet::SplitableDataSet(int inputs, int outputs, const SplitableDataSet* parent)
    : dim_input(inputs), dim_output(outputs), parent(parent)
{
    OPENANN_CHECK(inputs > 0);
    OPENANN_CHECK(outputs > 0);

    if(parent == 0) {
        std::srand(unsigned(std::time(0)));
    }
}

SplitableDataSet::~SplitableDataSet() 
{
    if(parent == 0) {
        for(int i = 0; i < data.size(); i++) {
            delete data[i].first;
            delete data[i].second;
        }

    }

    for(int i = 0; i < responsible_groups.size(); i++) {
        delete responsible_groups[i];
    }
}


int SplitableDataSet::samples() 
{
    return data.size();
}


int SplitableDataSet::inputs()
{
    return dim_input;
}


int SplitableDataSet::outputs()
{
    return dim_output;
}


Vt& SplitableDataSet::getInstance(int i)
{
    OPENANN_CHECK_WITHIN(i, 0, samples() - 1);
    return *(data.at(i).first);
}


Vt& SplitableDataSet::getTarget(int i)
{
    OPENANN_CHECK_WITHIN(i, 0, samples() - 1);
    return *(data.at(i).second);
}


void SplitableDataSet::finishIteration(Learner& learner)
{
}


void SplitableDataSet::add(Vt* instance, Vt* target)
{
    OPENANN_CHECK(parent == 0);

    data.push_back(std::pair<Vt*, Vt*>(instance, target));
}


void SplitableDataSet::add(Vt* instance, int klass)
{
    OPENANN_CHECK(parent == 0);

    Vt* target = new Vt(dim_output);
    target->setZero();

    if(klass > 0)
        (*target)(klass - 1) = 1.0;

    data.push_back(std::pair<Vt*, Vt*>(instance, target));
}


void SplitableDataSet::add(const instance_pair& pair)
{
    OPENANN_CHECK(parent == 0);

    data.push_back(pair);
}


void SplitableDataSet::split(std::vector<SplitableDataSet*>& groups, int number_of_groups)
{
    OPENANN_CHECK(number_of_groups > 1);

    int samples_per_group = data.size() / number_of_groups;
    int samples_in_group = 0;

    // allocate first group
    SplitableDataSet* current_set = new SplitableDataSet(dim_input, dim_output, this);

    for(int i = 0; i < data.size() && number_of_groups > 0; ++i) {
        current_set->add(data[i]);
        ++samples_in_group;

        if(samples_in_group >= samples_per_group) {
            // enough samples stored in this group. Generate a new one
            groups.push_back(current_set);
            responsible_groups.push_back(current_set);

            samples_in_group = 0;
            --number_of_groups;

            if(number_of_groups > 0)
                current_set = new SplitableDataSet(dim_input, dim_output, this);
        }
    }
}


void SplitableDataSet::split(std::vector<SplitableDataSet*>& groups, double ratio)
{
    OPENANN_CHECK_WITHIN(ratio, 0.0, 1.0);

    SplitableDataSet* first_set = new SplitableDataSet(dim_input, dim_output, this);
    SplitableDataSet* second_set = new SplitableDataSet(dim_input, dim_output, this);
    
    // number of samples for the first set
    int samples = std::floor(ratio * data.size() + 0.5);

    // share sample pointers on the two datasets depending on the ratio
    for(int i = 0; i < data.size(); ++i, --samples) {
        if(samples > 0)
            first_set->add(data[i]);
        else
            second_set->add(data[i]);
    }

    // add splitted sets to group vector
    groups.push_back(first_set);
    groups.push_back(second_set);

    // add splitted sets for memory management
    responsible_groups.push_back(first_set);
    responsible_groups.push_back(second_set);
}


SplitableDataSet* SplitableDataSet::merge(const std::vector<SplitableDataSet*>& groups) 
{
    OPENANN_CHECK(groups.size() > 0);

    SplitableDataSet* parent = const_cast<SplitableDataSet*>(groups.front()->parent);
    SplitableDataSet* merged_set = new SplitableDataSet(parent->inputs(), parent->outputs(), parent);

    for(int i = 0; i < groups.size(); ++i) {
        std::vector<instance_pair>::const_iterator beg = groups[i]->data.begin();
        std::vector<instance_pair>::const_iterator end = groups[i]->data.end();

        std::copy(beg, end, merged_set->data.end());
    }

    parent->responsible_groups.push_back(merged_set);

    return merged_set;
}


SplitableDataSet* SplitableDataSet::shuffle(int iteration)
{
    OPENANN_CHECK(iteration > 0);

    for(int i = 0; i < iteration; ++i) 
        std::random_shuffle(data.begin(), data.end(), default_random);

    return this;
}
 

} // namespace OpenANN
