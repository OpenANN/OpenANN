#pragma once

#include <vector>
#include "DataSet.h"

namespace OpenANN {

class Learner;

/**
 * @class SplitableDataSet
 *
 * A manipulable data set representation that can be splitted in several partitions
 *
 */


class SplitableDataSet : public DataSet
{
    typedef std::pair<Vt*, Vt*> instance_pair;

public:
    SplitableDataSet(int inputs, int outputs, const SplitableDataSet* parent = 0);
    virtual ~SplitableDataSet();

    virtual int samples();

    virtual int inputs();

    virtual int outputs();

    virtual Vt& getInstance(int i);

    virtual Vt& getTarget(int i);

    virtual void finishIteration(Learner& learner);
    
    virtual void add(Vt* instance, Vt* target);
    
    virtual void add(Vt* instance, int klass);

    virtual void add(const instance_pair& pair);

    virtual void shuffle(int iteration = 1);

    virtual void split(std::vector<SplitableDataSet*>& groups, int number_of_groups);

    virtual void split(std::vector<SplitableDataSet*>& groups, double ratio = 0.5);

    static SplitableDataSet& merge(const std::vector<SplitableDataSet*>& groups);
        
private:
    // vector dimensions 
    int dim_input;
    int dim_output;

    // data for all instances (features, target)
    std::vector<instance_pair> data;
    
    // for automatic memory management of instances
    const SplitableDataSet* parent;

    // for automatic memory management of splitted dataset groups
    std::vector<SplitableDataSet*> responsible_groups;
};

} // namespace OpenANN
