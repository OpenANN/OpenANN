class Constraint:
  NONE = 0
  DISTANCE = 1
  SLOPE = 2

cdef class Layer:
  cdef openann.Layer* construct(self, openann.OutputInfo& info):
    return NULL


cdef class SigmaPiLayer(Layer):
  cdef double std_dev
  cdef int bias
  cdef int width
  cdef int height
  cdef openann.ActivationFunction act
  cdef object nodes
  cdef openann.Constraint* distance
  cdef openann.Constraint* slope

  def __cinit__(self, width, height, activation, std_dev =0.05, bias=False):
    self.std_dev = std_dev
    self.bias = bias
    self.act = activation
    self.nodes = { 2: [], 3: [], 4: []}

    self.width = width
    self.height = height
    self.distance = new openann.DistanceConstraint(width, height)
    self.slope = new openann.SlopeConstraint(width, height)

  def __dealloc__(self):
    del self.distance
    del self.slope

  def add_second_order_nodes(self, numbers, constraint=Constraint.NONE):
    self.nodes[2].append( (numbers,constraint) )
    return self

  def add_third_order_nodes(self, numbers, constraint=Constraint.NONE):
    self.nodes[3].append((numbers, constraint)) 
    return self

  def add_fourth_order_nodes(self, numbers, constraint=Constraint.NONE):
    self.nodes[4].append((numbers, constraint)) 
    return self


  cdef openann.Layer* construct(self, openann.OutputInfo& info):
    cdef openann.SigmaPi* layer = new openann.SigmaPi(info, self.bias, self.act, self.std_dev)

    for lst in self.nodes[2]:
      if lst[1] == Constraint.DISTANCE:
        layer.secondOrderNodes(lst[0], deref(self.distance))
      elif lst[1] == Constraint.SLOPE:
        layer.secondOrderNodes(lst[0], deref(self.slope))
      else:
        layer.secondOrderNodes(lst[0])

    for lst in self.nodes[3]:
      if lst[1] == Constraint.DISTANCE:
        layer.thirdOrderNodes(lst[0], deref(self.distance))
      elif lst[1] == Constraint.SLOPE:
        layer.thirdOrderNodes(lst[0], deref(self.slope))
      else:
        layer.thirdOrderNodes(lst[0])

    for lst in self.nodes[4]:
      if lst[1] == Constraint.DISTANCE:
        layer.fourthOrderNodes(lst[0], deref(self.distance))
      elif lst[1] == Constraint.SLOPE:
        layer.fourthOrderNodes(lst[0], deref(self.slope))
      else:
        layer.fourthOrderNodes(lst[0])

    return layer








    




