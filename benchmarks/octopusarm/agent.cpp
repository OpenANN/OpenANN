#include "agent.h"
#include <OpenANN/OpenANN>
#include <OpenANN/optimization/IPOPCMAES.h>
#include <OpenANN/util/AssertionMacros.h>
#include <OpenANN/util/EigenWrapper.h>
#include <OpenANN/io/Logger.h>
#include <OpenANN/util/Random.h>
#include <ctime>
#include <fstream>
#include <stdlib.h>
#include <time.h>

using namespace OpenANN;

/**
 * \page OctopusArm Octopus Arm
 *
 * This is a reinforcement learning problem that has large dimensional state
 * and action space. Both are continuous, thus, we apply neuroevolution to
 * solve this problem.
 *
 * The octopus arm environment is available at <a href=
 * "http://www.cs.mcgill.ca/~dprecup/workshops/ICML06/octopus.html" target=
 * _blank>http://www.cs.mcgill.ca/~dprecup/workshops/ICML06/octopus.html</a>.
 * You have to unpack the archive octopus-code-distribution.zip to the working
 * directory.
 *
 * In this case, we only use the default environment settings. These are very
 * easy to learn. You see it in this picture:
 *
 * \image html octopusarm.png
 *
 * The octopus consists of 12 distinct compartments. It can move 36 muscles
 * and the state space has 106 components. The agent has to move the orange
 * pieces of food into the black mouth. We use an MLP with 106-10-36 topology
 * and bias. The action's components have to be in [0, 1]. Therefore, the
 * activation function of the output layer is logistic sigmoid. In the hidden
 * layer it is tangens hyperbolicus. In this benchmark we compare several
 * compression configurations. The weights of a neuron in the first layer are
 * represented by varying numbers of parameters (5-107) and the weights of a
 * neuron in the second layer are represented by 11 parameters.
 *
 * You can start the benchmark with
\verbatim
ruby run
\endverbatim
 * and evaluate the results with
\verbatim
ruby evaluate
\endverbatim
 * The log files that are needed for the evaluation script are collected in
 * the folder "logs" in the working directory. The evaluation script will list
 * the average return and the maximal return for each run and will calculate
 * the mean and standard deviation of the average returns and the maximum of
 * the maximal returns for each configuration. The average return indicates
 * how fast the agent learns a good policy and the maximal returns indicates
 * how good the best representable policy is.
 *
 * If you run this benchmark on one computer, it takes about 20 days. Thus,
 * it is recommended to start the benchmark on multiple computers. You can
 * modify the variable "runs" in the ruby script "run" and set it to a desired
 * number, start the script on separate computers, merge the results in a
 * single directory "logs" and run the script "evaluate". Each run will take
 * approximately two days. If you set the number of runs to 2 and run the
 * script on 5 computers, it will take about 4 days to finish.
 */

int num_states = 0, num_actions = 0;
IPOPCMAES opt;
Net net;
double episodeReturn;
Logger logger(Logger::CONSOLE);
int hiddenUnits;
int parameters;
double bestReturn;
Vt bestParameters;

int agent_init(int num_state_variables, int num_action_variables, int argc, const char *agent_param[])
{
  num_states = num_state_variables;
  num_actions = num_action_variables;

  parameters = 0;
  hiddenUnits = 10;
  if(argc > 0)
    parameters = atoi(agent_param[0]);
  if(argc > 1)
    hiddenUnits = atoi(agent_param[1]);

  net.inputLayer(num_states);
  if(parameters > 0)
  {
    net.compressedLayer(hiddenUnits, parameters, TANH, "dct");
    net.compressedOutputLayer(num_actions, hiddenUnits+1, LOGISTIC, "dct");
  }
  else
  {
    net.fullyConnectedLayer(hiddenUnits, TANH);
    net.outputLayer(num_actions, LOGISTIC);
  }
  bestParameters = net.currentParameters();
  bestReturn = -std::numeric_limits<double>::max();

  StoppingCriteria stop;
  stop.maximalFunctionEvaluations = 5000;
  stop.maximalRestarts = 1000;
  opt.setOptimizable(net);
  opt.setStopCriteria(stop);
  opt.restart();

  logger << net.dimension() << " parameters, " << num_states
      << " state components, " << num_actions << " action components\n";
  return 0;
}

const char* agent_get_name()
{
  std::stringstream stream;
  stream << "Neuroevolution_h_" << hiddenUnits << "_p_" << parameters;
  return stream.str().c_str();
}

Vt convert(double state[])
{
  Vt s(num_states);
  for(int i = 0; i < num_states; i++)
    s(i) = state[i];
  return s;
}

void convert(const Vt& action, double* out)
{
  for(int i = 0; i < num_actions; i++)
    out[i] = action(i);
}

int chooseAction(double state_data[], double out_action[])
{
  Vt state = convert(state_data);
  OPENANN_CHECK_MATRIX_BROKEN(state);
  Vt y = net(state);
  Vt action(num_actions);

  action = y;
  if(isMatrixBroken(action))
    action.fill(0.0);

  convert(action, out_action);
  return 0;
}

int agent_start(double state_data[], double out_action[])
{
  net.setParameters(opt.getNext());
  episodeReturn = 0;
  chooseAction(state_data, out_action);
  return 0;
}

int agent_step(double state_data[], double reward, double out_action[])
{
  chooseAction(state_data, out_action);
  episodeReturn += reward;
  return  0;
}

int agent_end(double reward) {
  episodeReturn += reward;
  logger << "agend end, return = " << episodeReturn << "\n";
  if(episodeReturn > bestReturn)
  {
    bestReturn = episodeReturn;
    bestParameters = net.currentParameters();
  }
  RandomNumberGenerator rng;
  opt.setError(-episodeReturn+0.1*episodeReturn*rng.sampleNormalDistribution<double>());
  if(opt.terminated())
    opt.restart();
  return 0;
}

void agent_cleanup()
{
  time_t rawtime;
  struct tm* timeinfo;
  std::time(&rawtime);
  timeinfo = std::localtime(&rawtime);
  std::ofstream result((std::string(std::asctime(timeinfo)).substr(0, 24) + "-best.log").c_str());
  result << "Best Return = " << bestReturn << std::endl;
  result << "Hidden Units = " << hiddenUnits << std::endl;
  result << "Parameters = " << parameters << std::endl;
  result << "Best Parameters = " << std::endl << bestParameters << std::endl;
  result.close();
}
