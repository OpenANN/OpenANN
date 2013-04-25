cdef class StoppingCriteria:
  cdef openann.StoppingCriteria *thisptr
  
  def __cinit__(self, stop={}):
    self.thisptr = new openann.StoppingCriteria()
    self.__configure_stopping_criteria__(stop)

  def __dealloc__(self):
    del self.thisptr

  def __configure_stopping_criteria__(self, stop):
    self.thisptr.maximalIterations = stop.get('maximal_iterations', self.thisptr.maximalIterations)
    self.thisptr.maximalFunctionEvaluations = stop.get('maximal_function_evaluations', self.thisptr.maximalFunctionEvaluations)
    self.thisptr.maximalRestarts = stop.get('maximal_restarts', self.thisptr.maximalRestarts)
    self.thisptr.minimalValue = stop.get('minimal_value', self.thisptr.minimalValue)
    self.thisptr.minimalValueDifferences = stop.get('minimal_value_differences', self.thisptr.minimalValueDifferences)
    self.thisptr.minimalSearchSpaceStep = stop.get('minimal_search_space_step', self.thisptr.minimalSearchSpaceStep)


cdef class StochasticGradientDescent:
  cdef openann.MBSGD *thisptr

  def __cinit__(self, stop={}, learning_rate=0.01, momentum=0.5, batch_size=10, gamma=0.0, 
    learning_rate_decay=1.0, min_learning_rate=0.0, momentum_gain=.0, max_momentum=1.0, min_gain=1.0, max_gain=1.0):

    self.thisptr = new openann.MBSGD(learning_rate, momentum, batch_size, gamma, learning_rate_decay, min_learning_rate, momentum_gain, max_momentum, min_gain, max_gain)
    self.stopping_criteria = StoppingCriteria(stop)
    self.thisptr.setStopCriteria(deref((<StoppingCriteria>self.stopping_criteria).thisptr))

  def __str__(self):
    return self.thisptr.name().c_str()

  def __dealloc__(self):
    del self.thisptr

  def optimize(self, net, inputs, outputs):
    cdef openann.MatrixXd *training_input_matrix = __matrix_numpy_to_eigen__(inputs)
    cdef openann.MatrixXd *training_output_matrix = __matrix_numpy_to_eigen__(outputs)
    (<Net>net).thisptr.trainingSet(deref(training_input_matrix), deref(training_output_matrix))
    self.thisptr.setOptimizable(deref((<Net>net).thisptr))
    self.thisptr.optimize()
    del training_input_matrix
    del training_output_matrix


cdef class LMA:
  cdef openann.LMA *thisptr

  def __cinit__(self, stop={}):
    self.thisptr = new openann.LMA()
    self.stopping_criteria = StoppingCriteria(stop)
    self.thisptr.setStopCriteria(deref((<StoppingCriteria>self.stopping_criteria).thisptr))

  def __dealloc__(self):
    del self.thisptr

  def __str__(self):
    return self.thisptr.name().c_str()

  def optimize(self, net, inputs, outputs):
    cdef openann.MatrixXd *training_input_matrix = __matrix_numpy_to_eigen__(inputs)
    cdef openann.MatrixXd *training_output_matrix = __matrix_numpy_to_eigen__(outputs)
    (<Net>net).thisptr.trainingSet(deref(training_input_matrix), deref(training_output_matrix))
    self.thisptr.setOptimizable(deref((<Net>net).thisptr))
    self.thisptr.optimize()
    del training_input_matrix
    del training_output_matrix



