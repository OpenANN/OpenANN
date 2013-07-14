cdef class Layer:
  """Common base of all layers."""
  cdef openann.Layer* construct(self):
    return NULL


cdef class SigmaPiLayer(Layer):
  """Fully connected higher-order layer."""
  cdef int width
  cdef int height

  cdef openann.Constraint* distance
  cdef openann.Constraint* slope
  cdef openann.Constraint* triangle
    
  cdef openann.SigmaPi* layer

  def __cinit__(self, net, activation, std_dev=0.05, bias=False):
    cdef openann.OutputInfo info = \
        (<Net?>net).output_info((<Net?>net).number_of_layers()-1)
    self.layer = new openann.SigmaPi(info, bias, activation, std_dev)

    self.width = info.dimensions[1]
    self.height = info.dimensions[2]
    self.distance = NULL
    self.slope = NULL
    self.triangle = NULL

  def __dealloc__(self):
    if self.distance != NULL:
      del self.distance
    if self.slope != NULL:
      del self.slope
    if self.triangle != NULL:
      del self.triangle

  def distance_2nd_order_nodes(self, numbers):
    if self.distance == NULL:
      self.distance = new openann.DistanceConstraint(self.width, self.height)
    self.layer.secondOrderNodes(numbers, deref(self.distance))
    return self

  def slope_2nd_order_nodes(self, numbers):
    if self.slope == NULL:
      self.slope = new openann.SlopeConstraint(self.width, self.height)
    self.layer.secondOrderNodes(numbers, deref(self.slope))
    return self

  def triangle_3rd_order_nodes(self, numbers, resolution):
    if self.triangle == NULL:
      self.triangle = new openann.TriangleConstraint(self.width, self.height,
                                                     resolution)
    self.layer.thirdOrderNodes(numbers, deref(self.triangle))
    return self

  def second_order_nodes(self, numbers):
    self.layer.secondOrderNodes(numbers)
    return self

  def third_order_nodes(self, numbers):
    self.layer.thirdOrderNodes(numbers)
    return self

  cdef openann.Layer* construct(self):
    return self.layer

