
cdef class StochasticGradientDescent:
  cdef openann.MBSGD *thisptr
  cdef openann.StoppingCriteria *stopping_criteria

  def __cinit__(self, net, stop=None, learning_rate=0.01, momentum=0.5, batch_size=10, gamma=0.0, 
      learning_rate_decay=1.0, min_learning_rate=0.0, momentum_gain=.0, max_momentum=1.0, min_gain=1.0, max_gain=1.0):

    assert(net is not None, "Mini-Batch Stochastic Gradient Descent needs an optimizable object for net")

    self.thisptr = new openann.MBSGD(learning_rate, momentum, batch_size, gamma, learning_rate_decay, min_learning_rate, momentum_gain, max_momentum, min_gain, max_gain)
    self.stopping_criteria = new openann.StoppingCriteria()

    self.__configure_stopping_criteria__(stop)

    self.thisptr.setStopCriteria(deref(self.stopping_criteria))
    self.thisptr.setOptimizable(deref((<Net>net).thisptr))

  def __configure_stopping_criteria__(self, stop):
    self.stopping_criteria.maximalIterations = stop.get('maximal_iterations', self.stopping_criteria.maximalIterations)
    self.stopping_criteria.maximalFunctionEvaluations = stop.get('maximal_function_evaluations', self.stopping_criteria.maximalFunctionEvaluations)
    self.stopping_criteria.maximalRestarts = stop.get('maximal_restarts', self.stopping_criteria.maximalRestarts)
    self.stopping_criteria.minimalValue = stop.get('minimal_value', self.stopping_criteria.minimalValue)
    self.stopping_criteria.minimalValueDifferences = stop.get('minimal_value_differences', self.stopping_criteria.minimalValueDifferences)
    self.stopping_criteria.minimalSearchSpaceStep = stop.get('minimal_search_space_step', self.stopping_criteria.minimalSearchSpaceStep)


  def __str__(self):
    return self.thisptr.name().c_str()

  def __dealloc__(self):
    del self.thisptr
    del self.stopping_criteria



  def optimize(self):
    self.thisptr.optimize()


  def step(self):
    return self.thisptr.step()



