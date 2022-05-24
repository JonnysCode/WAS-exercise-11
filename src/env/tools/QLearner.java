package tools;

import java.util.*;
import java.util.logging.*;
import cartago.Artifact;
import cartago.OPERATION;
import cartago.OpFeedbackParam;

public class QLearner extends Artifact {

  private Lab lab;
  private int stateCount;
  private int actionCount;
  private HashMap<Integer, double[][]> qTables;
  private Random random;

  private static final Logger LOGGER = Logger.getLogger(QLearner.class.getName());

  public void init(String environmentURL) {

    this.lab = new Lab(environmentURL);

    this.stateCount = this.lab.getStateCount();
    LOGGER.info("Initialized with a state space of n=" + stateCount);

    this.actionCount = this.lab.getActionCount();
    LOGGER.info("Initialized with an action space of m=" + actionCount);

    qTables = new HashMap<>();
    random = new Random();
  }

  /**
   * Computes a Q matrix for the state space and action space of the lab, and
   * against
   * a goal description. For example, the goal description can be of the form
   * [z1level, z2Level],
   * where z1Level is the desired value of the light level in Zone 1 of the lab,
   * and z2Level is the desired value of the light level in Zone 2 of the lab.
   * For exercise 11, the possible goal descriptions are:
   * [1,1], [1,2], [1,3], [2,1], [2,2], [2,3], [3,1], [3,2], [3,3].
   *
   * <p>
   * HINT 1: Use the methods of {@link LearningEnvironment} (implemented in
   * {@link Lab})
   * to interact with the learning environment (here, the lab), e.g., to retrieve
   * the
   * applicable actions, perform an action at the lab etc.
   * </p>
   * <p>
   * HINT 2: Use the method {@link #initializeQTable()} to retrieve an initialized
   * Q matrix.
   * </p>
   * <p>
   * HINT 3: Use the method {@link #printQTable(double[][])} to print a Q matrix.
   * </p>
   * 
   * @param goalDescription the desired goal against the which the Q matrix is
   *                        calculated (e.g., [2,3])
   * @param episodes        the number of episodes used for calculating the Q
   *                        matrix
   * @param alpha           the learning rate with range [0,1].
   * @param gamma           the discount factor [0,1]
   * @param epsilon         the exploration probability [0,1]
   * @param reward          the reward assigned when reaching the goal state
   */
  @OPERATION
  public void calculateQ(Object[] goalDescription, Object episodes, Object alpha, Object gamma, Object epsilon,
      Object reward) {

    double[][] qTable = initializeQTable();

    for (int i = 0; i < (int) episodes; i++) {

      // TODO: For all possible goal states ???

      while (!goalReached(goalDescription)) {
        int state = lab.readCurrentState();
        int action = epsilonGreedyStrategy(state, qTable, (double) epsilon);

        lab.performAction(action);
        double actionReward = getActionReward((double) reward, goalReached(goalDescription), action);

        qTable[state][action] = qNew(qTable, (double) alpha, actionReward, (double) gamma, state, action);
      }
    }

    qTables.put(goalKey(goalDescription), qTable);
  }

  /**
   * 
   * @param goalDescription  desired lab state (e.g., [2,3])
   * @param stateDescription current state of the lab (e.g.,
   *                         [2,2,true,false,true,true,2])
   */
  @OPERATION
  public void getActionFromState(Object[] goalDescription, Object[] stateDescription, OpFeedbackParam<String> actionTag,
      OpFeedbackParam<Object[]> payloadTags, OpFeedbackParam<Object[]> payload) {
    double[][] qTable = qTables.get(goalKey(goalDescription));

    List<Integer> stateDesc = Arrays.asList(stateDescription).stream().map(o -> (int) o).toList();
    int state = new ArrayList<>(lab.stateSpace).indexOf(stateDesc);

    int actionIndex = bestAction(lab.getApplicableActions(state), qTable[state]);

    Action action = lab.getAction(actionIndex);
    actionTag.set(action.getActionTag());
    payloadTags.set(action.getPayloadTags());
    payload.set(action.getPayload());
  }

  private int goalKey(Object[] goalDescription) {
    return (int) goalDescription[0] * 10 + (int) goalDescription[1];
  }

  private double qNew(double[][] qTable, double alpha, double reward, double gamma, int state, int action) {
    return qTable[state][action] + alpha * (reward + gamma * maxQ(qTable) - qTable[state][action]);
  }

  public double maxQ(double[][] qTable) {
    int state = lab.readCurrentState();
    List<Integer> actions = lab.getApplicableActions(state);

    double maxQ = Double.MIN_VALUE;
    for (int action : actions) {
      maxQ = qTable[state][action] > maxQ ? qTable[state][action] : maxQ;
    }
    return maxQ;
  }

  private double getActionReward(double goalReward, boolean goalReached, int action) {
    double actionReward = goalReached ? goalReward : 0.0;

    int stateAxis = lab.getAction(action).getApplicableOnStateAxis();
    if (stateAxis == 2 || stateAxis == 3) {
      actionReward += -50.0; // Switch lights ON/OFF
    } else if (stateAxis == 4 || stateAxis == 5) {
      actionReward += -1.0; // Move blinds UP/DOWN
    }

    return actionReward;
  }

  private int epsilonGreedyStrategy(int state, double[][] qTable, double epsilon) {

    List<Integer> actions = lab.getApplicableActions(state);
    int action = actions.get(0);

    if (random.nextDouble() < epsilon) {
      action = actions.get(random.nextInt(actions.size()));
    } else {
      action = bestAction(actions, qTable[state]);
    }

    return action;
  }

  private int bestAction(List<Integer> actions, double[] actionValues) {
    int action = actions.get(0);
    double max = Double.MIN_VALUE;

    for (Integer a : actions) {
      if (actionValues[a] > max) {
        max = actionValues[a];
        action = a;
      }
    }

    return action;
  }

  private boolean goalReached(Object[] goalDescription) {
    return Arrays.equals(lab.currentState.subList(0, 2).toArray(), goalDescription);
  }

  /**
   * Print the Q matrix
   *
   * @param qTable the Q matrix
   */
  void printQTable(double[][] qTable) {
    System.out.println("Q matrix");
    for (int i = 0; i < qTable.length; i++) {
      System.out.print("From state " + i + ":  ");
      for (int j = 0; j < qTable[i].length; j++) {
        System.out.printf("%6.2f ", (qTable[i][j]));
      }
      System.out.println();
    }
  }

  /**
   * Initialize a Q matrix
   *
   * @return the Q matrix
   */
  private double[][] initializeQTable() {
    double[][] qTable = new double[this.stateCount][this.actionCount];
    for (int i = 0; i < stateCount; i++) {
      for (int j = 0; j < actionCount; j++) {
        qTable[i][j] = 0.0;
      }
    }
    return qTable;
  }

  public static void main(String[] args) {
    QLearner learner = new QLearner();

    learner.init("https://raw.githubusercontent.com/Interactions-HSG/example-tds/was/tds/interactions-lab.ttl");

    learner.calculateQ(new Object[] { 2, 3 }, 10, 0.1, 0.5, 0.2, 100);
  }
}
