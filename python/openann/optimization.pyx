cdef class StoppingCriteria:
  """Stopping criteria for optimization algorithms."""
  cdef cbindings.StoppingCriteria *thisptr
  
  def __cinit__(self, stop={}):
    self.thisptr = new cbindings.StoppingCriteria()
    self.__configure_stopping_criteria__(stop)

  def __dealloc__(self):
    del self.thisptr

  def __configure_stopping_criteria__(self, stop):
    available_keys = ['maximal_iterations', 'maximal_function_evaluations',
                      'maximal_restarts', 'minimal_value',
                      'minimal_value_differences',
                      'minimal_search_space_step']
    unrecognized_keys = [k for k in stop.keys() if k not in available_keys]
    if len(unrecognized_keys) > 0:
      warnings.warn("Unrecognized stopping criteria: " +
                    str(unrecognized_keys))

    self.thisptr.maximalIterations = \
      stop.get('maximal_iterations', self.thisptr.maximalIterations)
    self.thisptr.maximalFunctionEvaluations = \
      stop.get('maximal_function_evaluations',
               self.thisptr.maximalFunctionEvaluations)
    self.thisptr.maximalRestarts = \
      stop.get('maximal_restarts', self.thisptr.maximalRestarts)
    self.thisptr.minimalValue = \
      stop.get('minimal_value', self.thisptr.minimalValue)
    self.thisptr.minimalValueDifferences = \
      stop.get('minimal_value_differences',
               self.thisptr.minimalValueDifferences)
    self.thisptr.minimalSearchSpaceStep = \
      stop.get('minimal_search_space_step',
               self.thisptr.minimalSearchSpaceStep)

cdef class Optimizer:
  """Common base of optimization algorithms."""
  cdef cbindings.Optimizer *thisptr
  cdef object stopping_criteria

cdef class MBSGD(Optimizer):
  """Mini-batch stochastic gradient descent."""
  def __cinit__(self,
      object stop={},
      learning_rate=0.01,
      momentum=0.5,
      batch_size=10,
      nesterov=False,
      learning_rate_decay=1.0,
      min_learning_rate=0.0,
      momentum_gain=0.0,
      max_momentum=1.0,
      min_gain=1.0,
      max_gain=1.0):

    self.thisptr = new cbindings.MBSGD(learning_rate, momentum, batch_size,
                                       nesterov, learning_rate_decay,
                                       min_learning_rate, momentum_gain,
                                       max_momentum, min_gain, max_gain)
    self.stopping_criteria = StoppingCriteria(stop)
    self.thisptr.setStopCriteria(deref((<StoppingCriteria>self.stopping_criteria).thisptr))

  def __str__(self):
    return self.thisptr.name().c_str()

  def __dealloc__(self):
    del self.thisptr

  def optimize(self, net, dataset):
    """Perform optimization until stopping criteria are satisfied."""
    (<Net?>net).thisptr.trainingSet(deref((<Dataset?>dataset).storage))
    self.thisptr.setOptimizable(deref((<Net>net).thisptr))
    self.thisptr.optimize()


cdef class LMA(Optimizer):
  """Levenberg-Marquardt algorithm."""
  def __cinit__(self, stop={}):
    self.thisptr = new cbindings.LMA()
    self.stopping_criteria = StoppingCriteria(stop)
    self.thisptr.setStopCriteria(deref((<StoppingCriteria>self.stopping_criteria).thisptr))

  def __dealloc__(self):
    del self.thisptr

  def __str__(self):
    return self.thisptr.name().c_str()

  def optimize(self, net, dataset):
    """Perform optimization until stopping criteria are satisfied."""
    (<Net?>net).thisptr.trainingSet(deref((<Dataset?>dataset).storage))
    self.thisptr.setOptimizable(deref((<Net>net).thisptr))
    self.thisptr.optimize()


cdef class CG(Optimizer):
  """Conjugate gradient."""
  def __cinit__(self, stop={}):
    self.thisptr = new cbindings.CG()
    self.stopping_criteria = StoppingCriteria(stop)
    self.thisptr.setStopCriteria(deref((<StoppingCriteria>self.stopping_criteria).thisptr))

  def __dealloc__(self):
    del self.thisptr

  def __str__(self):
    return self.thisptr.name().c_str()

  def optimize(self, net, dataset):
    """Perform optimization until stopping criteria are satisfied."""
    (<Net?>net).thisptr.trainingSet(deref((<Dataset?>dataset).storage))
    self.thisptr.setOptimizable(deref((<Net>net).thisptr))
    self.thisptr.optimize()


cdef class LBFGS(Optimizer):
  """Limited storage Broyden-Fletcher-Goldfarb-Shanno."""
  def __cinit__(self, stop={}, m=10):
    self.thisptr = new cbindings.LBFGS(m)
    self.stopping_criteria = StoppingCriteria(stop)
    self.thisptr.setStopCriteria(deref((<StoppingCriteria>self.stopping_criteria).thisptr))

  def __dealloc__(self):
    del self.thisptr

  def __str__(self):
    return self.thisptr.name().c_str()

  def optimize(self, net, dataset):
    """Perform optimization until stopping criteria are satisfied."""
    (<Net?>net).thisptr.trainingSet(deref((<Dataset?>dataset).storage))
    self.thisptr.setOptimizable(deref((<Net>net).thisptr))
    self.thisptr.optimize()
