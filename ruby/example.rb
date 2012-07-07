require 'openann'

mlp = MLP.new

training_input = []
training_output = []
(0...100).each do |i|
  x = i.to_f * Math::PI / 100.0
  training_input << [x]
  training_output << [Math.sin(x)]
end

# mlp.no_bias
mlp.input 1
mlp.fully_connected_hidden_layer 3, "tanh", -1
mlp.output 1, "sse", "tanh", -1
mlp.training_set training_input, training_output
# mlp.test_set training_input, training_output
mlp.training "lma"
mlp.fit_to_diff 1e-8
# mlp.fit_to_error 1e-8
# puts mlp.parameters

puts "t".ljust(10) + "y"
(0...training_input.size).each do |i|
  y = mlp.value training_input[i]
  puts sprintf("%.2f", training_output[i].to_s).to_s.ljust(10) + sprintf("%.2f", y.to_s)
end
