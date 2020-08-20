# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'ExpectimaxAgent', second = 'ExpectimaxAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ExpectimaxAgent(CaptureAgent):

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    self.start = gameState.getAgentPosition(self.index)
    self.agents_to_consider_list = [self.index] + self.getOpponents(gameState) # for the expectiminimax tree, consider yourself, and the 2 opponents only
    self.num_agents_to_consider = len(self.agents_to_consider_list)
    self.depth = 2 # TODO: tune

  def chooseAction(self, gameState):
    """
    Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
    """
    actions = gameState.getLegalActions(self.index)

    # ending case for game win
    foodLeft = len(self.getFood(gameState).asList())
    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start, pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    # standard case
    agent_index = 0 # start with self

    successors = [gameState.generateSuccessor(self.agents_to_consider_list[agent_index], action) for action in actions]
    next_agent_index = (agent_index + 1) % len(self.agents_to_consider_list)
    action_values = [self.getValue(successor, self.depth, next_agent_index) for successor in successors]

    best_value = max(action_values)
    index_of_best_value = action_values.index(best_value)
    return gameState.getLegalActions(self.index)[index_of_best_value]

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor


  def getValue(self, state, depth, index):
    if self.agents_to_consider_list[index] == self.index:  # if this is you
      depth -= 1

    # base case
    if depth == 0 or state.isOver():
      return self.evaluationFunction(state)

    if self.agents_to_consider_list[index] == self.index: # if this is you
      return self.maxValue(state, depth, index)
    else: # if this is an opponent
      return self.expValue(state, depth, index)

  def maxValue(self, state, depth, index):
    legal_actions = state.getLegalActions(self.agents_to_consider_list[index])
    next_states = [state.generateSuccessor(self.agents_to_consider_list[index], action) for action in legal_actions]
    next_index = (index + 1) % len(self.agents_to_consider_list)
    action_values = [self.getValue(next_state, depth, next_index) for next_state in next_states]

    return max(action_values)

  def expValue(self, state, depth, index):
    legal_actions = state.getLegalActions(self.agents_to_consider_list[index])
    next_states = [state.generateSuccessor(self.agents_to_consider_list[index], action) for action in legal_actions]
    next_index = (index + 1) % len(self.agents_to_consider_list)
    action_values = [self.getValue(next_state, depth, next_index) for next_state in next_states]

    distribution = [1 / len(legal_actions) for action in legal_actions]
    return sum(p * value for p, value in zip(distribution, action_values))

  def evaluationFunction(self, gameState):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState)
    weights = self.getWeights(gameState)
    return features * weights

  def getFeatures(self, gameState):
    # THIS FUNCTION EVALUATES A GAMESTATE GIVEN THE CURRENT STATE AND THE ACTION THAT COULD BE TAKEN
    # so it is basically evaluating the successor state of the agent

    ### if you would like it to evaluate any state passed in instead, just change the parameters and successor
        ### change it to only one parameter, named 'successor', and pass in the state you'd like to getFeatures of

    features = util.Counter()
    features['currScore'] = gameState.getScore()

    # my AgentState & position
    myState = gameState.getAgentState(self.index)
    myPos = myState.getPosition()

    # Determine if you are Pacman; if you are, isPacman = 1
    features['isPacman'] = 0
    if myState.isPacman: features['isPacman'] = 1

    # Get the successor score of whatever the action is, based on eating food
    foodList = self.getFood(gameState).asList()
    features['successorScore'] = -len(foodList)

    # Compute distance to the nearest food
    if len(foodList) > 0:  # copied from baselineTeam.py
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    # Compute distance to nearest capsule
    capsuleList = self.getCapsules(gameState)
    features['numCapsulesLeft'] = len(capsuleList)
    if features['numCapsulesLeft'] > 0:
      minDistance = min([self.getMazeDistance(myPos, food) for food in capsuleList])
      features['distanceToCapsule'] = minDistance

    # Determine how much food you have stored in your body
    foodCarried = myState.numCarrying
    # Determine how far you are from your original side (only relevant if you're carrying food)
    halfway = self.getFood(gameState).width // 2
    x, y = myPos
    if myState.isPacman and x > halfway:
        features['depositFood'] = foodCarried * abs(x - (halfway + 1))
    else:
        features['depositFood'] = foodCarried * abs(x - halfway)

    features['enemyGhosts'] = 0
    features['numScaredGhosts'] = 0
    features['enemyPacman'] = 0
    features['numEatPacman'] = 0


    # Determine distances to all enemy ghosts, add to feature if they are scared, subtract if not
    enemyIndexList = self.getOpponents(gameState)
    enemyList = [gameState.getAgentState(index) for index in enemyIndexList]
    # if they are enemy ghosts are they scared
    # if they are not scared, and you are Pacman, please run away
    for enemy in enemyList:
        if not enemy.isPacman:
            if enemy.scaredTimer > 0:
                features['enemyGhosts'] += self.getMazeDistance(myPos, enemy.getPosition())
                features['numScaredGhosts'] += 1
            else:
                features['enemyGhosts'] -= self.getMazeDistance(myPos, enemy.getPosition())
        else:
            features['enemyPacman'] += self.getMazeDistance(myPos, enemy.getPosition())
            features['numEatPacman'] += 1
            # if they are pacman, and you are ghost, go attack them: if you're scared don't, so apply negative
            if myState.scaredTimer > 0:
                features['enemyPacman'] = -features['enemyPacman']
    teamIndexList = self.getTeam(gameState)
    teamList = [gameState.getAgentState(index) for index in teamIndexList]

    return features

  def getWeights(self, gameState):
    return {'isPacman': 0, 'successorScore': 10000, 'distanceToFood': -100,
            'distanceToCapsule': -120, 'numCapsulesLeft': -15000,
            'depositFood': -5, 'numScaredGhosts': -70, 'enemyGhosts': -105,
            'numEatPacman': -190, 'enemyPacman': -100, 'currScore': 1}
