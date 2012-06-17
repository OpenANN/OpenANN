#ifndef AGENT_H
#define AGENT_H

/*
 * If any of these functions return < 0, it indicates an error and the agent 
 * handler will abort. A return value >= 0 indicates that the agent can 
 * continue
 */

/* The first two variables indicate the size of the state_data and
 * out_action arrays respectively, these numbers are constant throughout
 * the length of the simulation.
 *
 * argc and agent_param are the remainder of the argv parameters passed on the
 * command line minus the first 4 (name of program, ip and port of
 * server and number of episodes)
 */
int agent_init(int num_state_variables, int num_action_variables, int argc, const char *agent_param[]);

/* When this function is called, the agent should return it's name. This
 * name will be used to identify individual agents during the competition.
 * The name should not contain any spaces nor linefeed characters.
 */
const char* agent_get_name();

/* Start an episode.
 * The agent is given the initial state in state_data.
 * The agent gives the action to take in out_action.
 * The size of these arrays has been specified in a prior call to agent_init.
 */
int agent_start(double state_data[], double out_action[]);

/* Perform a single step of an episode.
 * The agent is given the reward for the previous action in reward.
 * The new state is given in state_data.
 * The agent gives the action to take in out_action.
 * The size of these arrays has been specified in a prior call to agent_init.
 */
int agent_step(double state_data[], double reward, double out_action[]);

/* The agent has reached a terminal state, indicating the end of an episode.
 * The agent is given the final reward for the previous action in reward.
 */
int agent_end(double reward);

/* The program will close so the agent should free any resource it has allocated
 */
void agent_cleanup();

#endif
