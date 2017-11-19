#########################################
#Modified/Added Keywords:
#1.searchUnexplored
#2.uniformCostSearch
#########################################

from game import Agent
from game import Actions
from game import Directions
import random
from util import manhattanDistance
import util
from submission import randomAgent
from submission import betterEvaluationFunction
from submission import absoluteDistance
from submission import rlAlgo
import copy
import collections

class searchUnexplored(object):
    def __init__(self, start,goal,wall,capsules,isOracle=False):
        self.start = start
        self.goal = goal
        self.wall=wall
        self.capsules=capsules
        self.isOracle=isOracle
    def startState(self):
        return (self.start[0],self.start[1])
    def isEnd(self, state):
        state = (int(state[0]), int(state[1]))
        if self.goal[state[0]][state[1]]:
          return True
    def succAndCost(self, state):
        result = []
        x=int(state[0])
        y=int(state[1])
        
        if self.isOracle:
            for delta in [+1,-1]:
                if not (self.wall[x+delta][y] or ((x+delta, y) in self.capsules and not self.goal[x+delta][y])):
                    result.append(((delta,0),(x+delta,y),1))
                if not (self.wall[x][y+delta] or ((x, y+delta) in self.capsules and not self.goal[x][y+delta])):
                    result.append(((0,delta),(x,y+delta),1))
        else:
            for delta in [+1,-1]:
                if not self.wall[x+delta][y]:
                    result.append(((delta,0),(x+delta,y),1))
                if not self.wall[x][y+delta]:
                    result.append(((0,delta),(x,y+delta),1))
        return result

def uniformCostSearch(problem):
    front = util.PriorityQueue()
    front.push((problem.startState(),0), 0)
    back = set()
    #implement command?
    viewRange=float('inf')
    while not front.isEmpty():
        state, pastCost = front.pop()
        if pastCost > viewRange:
          break
        if problem.isEnd(state):
            return pastCost
        for action, newState, cost in problem.succAndCost(state):
            if newState not in back:
              front.push((newState, pastCost + cost),pastCost + cost)
              back.add(newState)
    return float('inf')

def rlReward(index, gameState):
    selfPosition=gameState.getGhostPosition(index)
    agentsPositions=[gameState.getPacmanPosition()]+gameState.getGhostPositions()
    agentsPositions.remove(selfPosition)
    # print "I am at ",selfPosition ,";where are other agents? ", agentsPositions
    #####Sum of the distance from all other agents
    # sumDistance=0.0
    # for otherPosition in agentsPositions:
    #     sumDistance+=absoluteDistance(otherPosition,selfPosition)
    # print "How far away are they? ", sumDistance

    #####Only calculate the nearest agent
    nearestAgent=float('inf')
    nearestAgentPos=()
    for otherPosition in agentsPositions:
      temp=absoluteDistance(otherPosition,selfPosition)
      if  temp < nearestAgent:
        nearestAgent=temp
        nearestAgentPos=otherPosition

    food = gameState.getFood()
    wall = gameState.getWalls()
    capsules = gameState.getCapsules()
    UnexploredRegion = searchUnexplored(selfPosition,food,wall,capsules)
    nearestUnexplored= uniformCostSearch(UnexploredRegion)
    # if nearestUnexplored == float('inf'):
    #   if not gameState.isWin():
    #     return random.choice(gameState.getLegalActions(index)) 
    #   return 'Stop'
    
    curPacX, curPacY=selfPosition
    legalMoves=gameState.getLegalActions(index)
    random.shuffle(legalMoves)

    reward=collections.defaultdict(float)
    for nextAction in legalMoves:
      closerToUnexplored=0
      nextPacX, nextPacY = (0,0)
      if nextAction == 'East':
        nextPacX, nextPacY = (curPacX+1,curPacY)
      if nextAction == 'North':
        nextPacX, nextPacY = (curPacX,curPacY+1)
      if nextAction == 'South':
        nextPacX, nextPacY = (curPacX,curPacY-1)
      if nextAction == 'West':
        nextPacX, nextPacY = (curPacX-1,curPacY)
      # if gameState.getFood()[nextPacX][nextPacY]:
      #   return nextAction
      nextPacPos=(nextPacX,nextPacY)
      searchFood = searchUnexplored(nextPacPos,food,wall,capsules)
      
      nF= uniformCostSearch(searchFood)
      # rewardForCloseToUnexplored=2
      # if nF < nearestUnexplored:
        # return nextAction
        # closerToUnexplored=rewardForCloseToUnexplored
      reward[nextAction]+=2.71828**(-nF)

      weghitForHowFar=0.5
      #####Calculate the sum of distances
      # newSumDistance=0
      # for otherPosition in agentsPositions:
      #   newSumDistance+=absoluteDistance(otherPosition,nextPacPos)
      # if newSumDistance > sumDistance:
      #   rewardForGetAwayFromOthers=weghitForHowFar*(2.71828**(-newSumDistance))
      #   reward[nextAction]+=rewardForGetAwayFromOthers

      #####Calculate the nearest
      newNearestDistance=absoluteDistance(nextPacPos,nearestAgentPos)
      if newNearestDistance > nearestAgent:
        rewardForGetAwayFromOthers=weghitForHowFar*(2.71828**(-newNearestDistance))
        reward[nextAction]+=rewardForGetAwayFromOthers


    return reward

act2Vec = {'North':(0,1), 'East':(1,0), 'West':(-1,0), 'South':(0,-1)}
class GhostAgent( Agent ):
  def __init__( self, index ):
    self.index = index
    self.algo = rlAlgo
  ##RL
  """
  def getAction(self, gameState):
    if random.random() < self.algo.explorationProb:
      # print self
      action = random.choice(gameState.getLegalActions(self.index))
      return action
    else:
      reward=collections.defaultdict(float)  # map from actions to rewards
      # print "Ghost"
      # print self.algo.weights
      for nextAction in gameState.getLegalActions(self.index):
        reward[nextAction] = self.algo.getScoreForAction(gameState, nextAction, self.index)
      
      maxValue=max(reward.values())
      choices=[choice for choice in reward if reward[choice]==maxValue]
      action = random.choice(choices)
      return action
  """
  
  ##Oracle
  def filterOutCapsuleActions(self, actions, curPos, capsules):
      updatedActions = copy.deepcopy(actions)
      for act in actions:
        if act == "Stop": continue
        nextPos = (curPos[0]+act2Vec[act][0], curPos[1]+act2Vec[act][1])
        if nextPos in capsules:
            # print "ghost remove act:", act
            updatedActions.remove(act)
      return updatedActions
    
  def getAction( self, state ):
    # dist = self.getDistribution(state)
    # if len(dist) == 0: 
    #   return Directions.STOP
    # else:
    #   return util.chooseFromDistribution( dist )
    food = state.getFood()
    # print "====================Ghost food======================"
    # print food
    wall = state.getWalls()
    capsules = state.getCapsules()
    # for capX, capY in capsules:
    #   wall[capX][capY]=True

    # realWall=wall[:]
    # pacDir=state.getGhostState(self.index).getDirection()
    # print state
    # print "direction of ghost=",pacDir  

    curPacX, curPacY=state.getGhostPosition(self.index)
    curPacPos=(curPacX, curPacY)
    UnexploredRegion = searchUnexplored(curPacPos,food,wall,capsules,isOracle=True)
    nearestUnexplored= uniformCostSearch(UnexploredRegion)
    legalMoves=state.getLegalActions(self.index)
    legalMoves = self.filterOutCapsuleActions(legalMoves, curPacPos, capsules)
    if nearestUnexplored == float('inf'):
      if not state.isWin():
        return random.choice(legalMoves) 
      return 'Stop'
    
    random.shuffle(legalMoves)
    for nextAction in legalMoves:
      nextPacX, nextPacY = (0,0)
      if nextAction == 'East':
        nextPacX, nextPacY = (curPacX+1,curPacY)
      if nextAction == 'North':
        nextPacX, nextPacY = (curPacX,curPacY+1)
      if nextAction == 'South':
        nextPacX, nextPacY = (curPacX,curPacY-1)
      if nextAction == 'West':
        nextPacX, nextPacY = (curPacX-1,curPacY)
      if state.getFood()[int(nextPacX)][int(nextPacY)]:
        return nextAction
      nextPacPos=(nextPacX,nextPacY)
      searchFood = searchUnexplored(nextPacPos,food,wall,capsules,isOracle=True)
      nF= uniformCostSearch(searchFood)
      # print "ghost nF", nF
      if nF < nearestUnexplored:
        return nextAction
    return random.choice(legalMoves) 
  
  def getDistribution(self, state):
    "Returns a Counter encoding a distribution over actions from the provided state."
    util.raiseNotDefined()

class RandomGhost( GhostAgent ):
  "A ghost that chooses a legal action uniformly at random."
  def getDistribution( self, state ):
    dist = util.Counter()
    for a in state.getLegalActions( self.index ): dist[a] = 1.0
    dist.normalize()
    return dist

class DirectionalGhost( GhostAgent ):
  "A ghost that prefers to rush Pacman, or flee when scared."
  def __init__( self, index, prob_attack=0.8, prob_scaredFlee=0.8 ):
    self.index = index
    self.prob_attack = prob_attack
    self.prob_scaredFlee = prob_scaredFlee
      
  def getDistribution( self, state ):
    # Read variables from state
    ghostState = state.getGhostState( self.index )
    legalActions = state.getLegalActions( self.index )
    pos = state.getGhostPosition( self.index )
    isScared = ghostState.scaredTimer > 0
    
    speed = 1
    if isScared: speed = 0.5
    
    actionVectors = [Actions.directionToVector( a, speed ) for a in legalActions]
    newPositions = [( pos[0]+a[0], pos[1]+a[1] ) for a in actionVectors]
    pacmanPosition = state.getPacmanPosition()

    # Select best actions given the state
    distancesToPacman = [manhattanDistance( pos, pacmanPosition ) for pos in newPositions]
    if isScared:
      bestScore = max( distancesToPacman )
      bestProb = self.prob_scaredFlee
    else:
      bestScore = min( distancesToPacman )
      bestProb = self.prob_attack
    bestActions = [action for action, distance in zip( legalActions, distancesToPacman ) if distance == bestScore]
    
    # Construct distribution
    dist = util.Counter()
    for a in bestActions: dist[a] = bestProb / len(bestActions)
    for a in legalActions: dist[a] += ( 1-bestProb ) / len(legalActions)
    dist.normalize()
    return dist
