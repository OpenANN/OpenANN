/* Initialize agent by retrieving task information from server
 * and send agent name to server to start logging.
 */
int performInit(int sd, int argc, const char *agent_params[]);

/* Run the agent for an entire episode */
int performEpisode(int sd);

/* Calls the agent cleanup function */
void performCleanup();
