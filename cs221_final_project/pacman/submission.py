#########################################
#Modified/Added Keywords:
#1.randomAgent
# ==> don't use generateSuccessor anymore
#########################################

from util import manhattanDistance
from game import Directions
import random, util
import math
import collections
import copy

from game import Agent
def scoreEvaluationFunction(currentGameState):

  return currentGameState.getScore()

  
class MultiAgentSearchAgent(Agent):

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class searchTarget(object):
    def __init__(self, start,goal,wall):
        self.start = start
        self.goal = goal
        self.wall=wall
    def startState(self):
        return (self.start[0],self.start[1])
    def isEnd(self, state):
        if state == self.goal: 
          return True
    def succAndCost(self, state):
        result = []
        x=state[0]
        y=state[1]
        actions = [1,-1]
        for delta in actions:
            if not self.wall[x+delta][y]:
                result.append(((delta,0),(x+delta,y),1))
            if not self.wall[x][y+delta]:
                result.append(((0,delta),(x,y+delta),1))
        return result

class searchFoods(object):
    def __init__(self, start,goal,wall,capsules,isOracle=False):
        self.start = start
        self.goal = goal
        self.wall=wall
        self.capsules=capsules
        self.isOracle=isOracle
    def startState(self):
        return (self.start[0],self.start[1])
    def isEnd(self, state):
        if self.goal[int(state[0])][int(state[1])]:
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
def futureCost(newState, goal):
      return manhattanDistance(newState,goal)

def Astar(problem):
    front = util.PriorityQueue()
    front.push((problem.startState(),0), 0)
    back = set()

    while True:
        # Move state from frontier to explored
        state, pastCost = front.pop()
        #print state, pastCost
        if problem.isEnd(state):
            return pastCost
        # Expand the frontier
        for action, newState, cost in problem.succAndCost(state):
            if newState not in back:
              heuristic=futureCost(newState,problem.goal)
              front.push((newState, pastCost + cost),pastCost + cost+heuristic)
              back.add(newState)

def uniformCostSearch(problem):
    front = util.PriorityQueue()
    front.push((problem.startState(),0), 0)
    back = set()
    viewRange=float('inf')
    while not front.isEmpty():
        state, pastCost = front.pop()
        if pastCost > viewRange:
          break
        if problem.isEnd(state):
            return [state,pastCost]
        for action, newState, cost in problem.succAndCost(state):
            if newState not in back:
              front.push((newState, pastCost + cost),pastCost + cost)
              back.add(newState)
    return None,float('inf')

act2Vec = {'North':(0,1), 'East':(1,0), 'West':(-1,0), 'South':(0,-1)}
class oracleAgent(MultiAgentSearchAgent):

  def filterOutCapsuleActions(self, actions, curPos, capsules):
      updatedActions = copy.deepcopy(actions)
      for act in actions:
        if act == "Stop": continue
        nextPos = (curPos[0]+act2Vec[act][0], curPos[1]+act2Vec[act][1])
        if nextPos in capsules:
            updatedActions.remove(act)
      return updatedActions

  def getAction(self, gameState):
    
    # nearestUnexplored=betterEvaluationFunction(gameState, 0)
    searchFood = searchFoods(gameState.getPacmanPosition(),gameState.getFood(),gameState.getWalls(),gameState.getCapsules(),isOracle=True)  
    _, nearestUnexplored  = uniformCostSearch(searchFood)
    if nearestUnexplored == float('inf'):
      return 'Stop'
    food = gameState.getFood()
    # print '======================================'
    # print food
    wall = gameState.getWalls()
    capsules = gameState.getCapsules()
    # for capX, capY in capsules:
    #   wall[capX][capY]=True
    curPacX, curPacY=gameState.getPacmanPosition()
    legalMoves=gameState.getLegalActions()
    # print gameState
    # print "before: legalMoves: ", legalMoves
    legalMoves = self.filterOutCapsuleActions(legalMoves, (curPacX, curPacY), capsules)
    # print "legalMoves: ", legalMoves
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
      if gameState.getFood()[nextPacX][nextPacY]:
        return nextAction
      nextPacPos=(nextPacX,nextPacY)
      searchFood = searchFoods(nextPacPos,food,wall,capsules,isOracle=True)
      pos,nF= uniformCostSearch(searchFood)
      if nF < nearestUnexplored:
        return nextAction
    return random.choice(legalMoves)  


class randomAgent(MultiAgentSearchAgent):
  def getAction(self, gameState):

    
    nearestUnexplored=betterEvaluationFunction(gameState, 0)
    if nearestUnexplored == float('inf'):
      if not gameState.isWin():
        return random.choice(gameState.getLegalActions()) 
      return 'Stop'
    food = gameState.getFood()
    wall = gameState.getWalls()
    capsules = gameState.getCapsules()
    curPacX, curPacY=gameState.getPacmanPosition()
    legalMoves=gameState.getLegalActions()
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
      if gameState.getFood()[nextPacX][nextPacY]:
        return nextAction
      nextPacPos=(nextPacX,nextPacY)
      searchFood = searchFoods(nextPacPos,food,wall,capsules)
      pos,nF= uniformCostSearch(searchFood)
      if nF < nearestUnexplored:
        return nextAction
    return random.choice(gameState.getLegalActions())  


class naiveAgent(MultiAgentSearchAgent):   
  def getAction(self, gameState): 
    return random.choice(gameState.getLegalActions())

def betterEvaluationFunction(currentGameState,index):
  """
    Your extreme, unstoppable evaluation function (problem 4).

    DESCRIPTION: <write something here so we know what you did>
  """

  # BEGIN_YOUR_CODE (our solution is 26 lines of code, but don't worry if you deviate from this)
  food = currentGameState.getFood()
  row = len(list(food)[0])
  col = len(list(food))
  size=(col,row)
  capsules=currentGameState.getCapsules()
  ghostPos = currentGameState.getGhostPositions()
  GhostStates = currentGameState.getGhostStates()
  ScaredTimes=[ghostState.scaredTimer for ghostState in GhostStates]
  
  normalGhost=[]
  scaredGhost=[]
  for ghost in GhostStates:
    if ghost.scaredTimer > 0:
      scaredGhost.append(ghost.getPosition())
    else:
      normalGhost.append(ghost.getPosition())
  pos=()
  if index ==0:
    pos=currentGameState.getPacmanPosition()
  else:
    pos=currentGameState.getGhostPosition(index)
  wall=currentGameState.getWalls()
  
  class searchTarget(object):
    def __init__(self, start,goal,wall):
        self.start = start
        self.goal = goal
        self.wall=wall
    def startState(self):
        return (self.start[0],self.start[1])
    def isEnd(self, state):
        if (float(state[0]),float(state[1])) in self.goal or\
            (float(state[0])+0.5,float(state[1])) in self.goal or \
            (float(state[0]),float(state[1])+0.5) in self.goal or \
            (float(state[0])-0.5,float(state[1])) in self.goal or \
            (float(state[0]),float(state[1])-0.5) in self.goal: return True
    def succAndCost(self, state):
        result = []
        x=state[0]
        y=state[1]
        actions = [1,-1]
        for delta in actions:
            if not self.wall[x+delta][y]:
                result.append(((delta,0),(x+delta,y),1))
            if not self.wall[x][y+delta]:
                result.append(((0,delta),(x,y+delta),1))
        return result

  def futureCost(newState, goal):
      return sum(manhattanDistance(newState,pos) for pos in goal)/float(len(goal))     
  
  def uniformCostSearch(problem):
    front = util.PriorityQueue()
    front.push((problem.startState(),0), 0)
    back = set()
    while not front.isEmpty():
        state, pastCost = front.pop()
        if problem.isEnd(state):
            return pastCost
        for action, newState, cost in problem.succAndCost(state):
            if newState not in back:
              front.push((newState, pastCost + cost),pastCost + cost)
              back.add(newState)
    return float('inf')
  
  def Astar(problem):
    front = util.PriorityQueue()
    front.push((problem.startState(),0), 0)
    back = set()

    while True:
        # Move state from frontier to explored
        state, pastCost = front.pop()
        #print state, pastCost
        if problem.isEnd(state):
            return pastCost
        # Expand the frontier
        for action, newState, cost in problem.succAndCost(state):
            if newState not in back:
              heuristic=futureCost(newState,problem.goal)
              front.push((newState, pastCost + cost),pastCost + cost+heuristic)
              back.add(newState)
  
  searchFood = searchFoods(pos,food,wall,capsules)
  pos, nF= uniformCostSearch(searchFood)

  return nF

def absoluteDistance(pos1,pos2):
    return math.sqrt((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2)

class rlAlgo(object):
  
  #initial all weights to 1.0
  weights = collections.defaultdict(lambda:0.0)  # maps from feature key to its weight
  # weights['Closer?']=20.0
  # weights['Closer to other people?']= -10.0
  numIters = 0
  discount = 0.5
  explorationProb = 1
  # TODO: feature key includes agentIdx, this enables different agents to learn different weights
  @classmethod
  def getStepSize(self):
    return 1.0 / math.sqrt(self.numIters)
  # def startState(self, )
  @classmethod
  # def getScoreForAction(self, gameState, preCompData, action, agentIdx):
  def getScoreForAction(self, gameState, action, agentIdx):
    score = 0.0
    for f, v in self.featureExtractor(gameState, action, agentIdx):
      score += self.weights[f] * v
    return score
  @classmethod
  def numUnexplored(self, state):
    numUnexplored=0
    food=state.getFood()
    row=len(list(food)[0])
    col=len(list(food))
    for i in range(col):
      for j in range(row):
        if food[i][j]: numUnexplored+=1
    return numUnexplored

  @classmethod  
  def featureExtractor(self, gameState, nextAction, agentIdx):
    """output a feature list of (key, value) pairs"""
    featureList = []
    selfPosition=gameState.getPacmanPosition() if agentIdx ==0 \
                     else gameState.getGhostPosition(agentIdx)

    nextState=gameState.generateSuccessor(agentIdx,nextAction)
    nextPosition=nextState.getPacmanPosition() if agentIdx ==0 \
                     else nextState.getGhostPosition(agentIdx)    
    ############################
    #Feature:allAgentsPositions#
    ############################              
    allPositions=[gameState.getPacmanPosition()]+gameState.getGhostPositions()
    result = tuple([x for t in allPositions for x in t])
    # featureList.append(((nextAction, result), 1))

    ##############################################################################
    #Feature:Does this action bring the agent closer to the unexplored spot?     #
    ##############################################################################               
    searchFood = searchFoods(selfPosition,gameState.getFood(),gameState.getWalls(),gameState.getCapsules())  
    nearestUnexplored, distance = uniformCostSearch(searchFood)
    newDistance=0
    if nearestUnexplored != None:
      afterAction=searchTarget(nextPosition,nearestUnexplored,gameState.getWalls())
      newDistance=Astar(afterAction)
    x=selfPosition[0]
    y=selfPosition[1]
    if newDistance < distance:
      featureList.append(((nextAction,selfPosition,'Closer?'),1))
      featureList.append(('Closer?',1))
    else:
      featureList.append(((nextAction,selfPosition,'Closer?'),0))
      featureList.append(('Closer?',0))
    
    ####################################################################
    #Feature:Does this action bring the agent closer to another agent? #
    ####################################################################
    agentsPositions=[gameState.getPacmanPosition()]+gameState.getGhostPositions()
    agentsPositions.remove(selfPosition)
    nearestAgent=float('inf')
    nearestAgentPos=()
    newOtherAgentDistance=float('inf')
    for otherPosition in agentsPositions:
      temp=absoluteDistance(otherPosition,selfPosition)
      if  temp < nearestAgent:
        nearestAgent=temp
        newOtherAgentDistance=absoluteDistance(otherPosition,nextPosition)
        nearestAgentPos=otherPosition
    
    if newOtherAgentDistance < nearestAgent:
      featureList.append(('Closer to other people?',1))
    else:
      featureList.append(('Closer to other people?',0))
    
    ##################################################################
    #Feature: How to represent the distribution of unexploredRegion? #
    ##################################################################

    # unexploredRegion=self.numUnexplored(gameState)
    # unexploredRawData=gameState.getFood()
    # row=len(list(unexploredRawData)[0])
    # col=len(list(unexploredRawData))

    # xArray=[]
    # for i in range(1,col):
    #   count=0
    #   for j in range(1,row):
    #     if unexploredRawData[i][j]==True:
    #       count+=1
    #   if count==0:
    #     xArray.append(0)
    #     # featureList.append(((nextAction,'col',i,0),1))
    #   else:
    #     xArray.append(1)
    #     # featureList.append(((nextAction,'col',i,count),1))
    # # featureList.append(((nextAction,tuple(xArray)),1))

    # yArray=[]
    # for j in range(1,row):
    #   count=0
    #   for i in range(1,col):
    #     if unexploredRawData[i][j]==True:
    #       count+=1
    #   if count==0:
    #     yArray.append(0)
    #     # featureList.append(((nextAction,'row',j,0),1))
    #   else:
    #     yArray.append(1)
        # featureList.append(((nextAction,'row',j,count),1))
    
    # featureList.append(((nextAction,tuple(xArray+yArray)),1))

    # def compute(point, unexploredRawData, wall):
    #   count = 0
    #   x, y = point
    #   for i in [0,1]:
    #     for j in [0,1]:
    #       # if wall[x+i][y+j]:
    #       #   unexploredRawData[x+i][y+j]=True
    #       if unexploredRawData[x+i][y+j]:
            

    # # print gameState
    # count=0
    # for i in range(1,col,2):
    #   for j in range(1,row,2):
    #     point =(i,j)
    #     compute(point, array, unexploredRawData, wall)
        


    # print array
    # featureList.append(((nextAction,tuple(array)), 1))
    # unexploredRegion=self.numUnexplored(gameState)
    # unexploredRawData=gameState.getFood()
    # row=len(list(unexploredRawData)[0])
    # col=len(list(unexploredRawData))
    # # print gameState
    # array=[]
    # count=0
    # grid_size = 2
    # def checkIfUnexplored(i,j):
    #   for deltaX in range(grid_size):
    #       for deltaY in range(grid_size):
    #         if unexploredRawData[i+deltaX][j+deltaY]==True:
    #           return(((nextAction,(i/grid_size,j/grid_size), 1),1))
    #   return ()

    # tempFeatureList = []
    # for i in range(1,col,grid_size):
    #   for j in range(1,row,grid_size):
    #     if checkIfUnexplored(i,j):
    #       array.append(1)
    #       # tempFeatureList.append(checkIfUnexplored(i,j))
    #     else:
    #       array.append(0)
          # tempFeatureList.append(((nextAction,(i/grid_size,j/grid_size), 0),1))
    # featureList.append(((nextAction,tuple(array)),1))
    # add joint feature --> it turns out to be useless
    # fkey = tuple([(k[1], k[2]) for k, v in tempFeatureList] + [nextAction])
    # featureList += (tempFeatureList + [(fkey, 1)])
    # featureList += tempFeatureList



    return featureList

  @classmethod
  def incorporateFeedback(self, state, action, reward, newState, agentIndex):
    if not newState.isWin() or newState.isLose():
      eta = self.getStepSize()
      v_opt= max((self.getScoreForAction(newState, act, agentIndex)) \
                                      for act in newState.getLegalActions(agentIndex))
      qVal = self.getScoreForAction(state, action, agentIndex)
      for feature in self.featureExtractor(state, action, agentIndex):
        self.weights[feature[0]]=self.weights[feature[0]]-\
                                (eta*(qVal-(reward+self.discount*v_opt)))*feature[1]

class rlAgent(MultiAgentSearchAgent):

  def getAction(self, gameState):
    algo = rlAlgo
    algo.numIters += 1
    # print "Closer to unexplored? :", algo.weights['Closer?']
    # print "Closer to another agent? :",algo.weights['Closer to other people?']
    capsules=gameState.getCapsules()

    if random.random() < algo.explorationProb:
      action=random.choice(gameState.getLegalActions(self.index))
      return action
    else:
      reward=collections.defaultdict(float)  # map from actions to rewards
      for action in gameState.getLegalActions(self.index):
        reward[action] = algo.getScoreForAction(gameState, action, self.index)
      maxValue=max(reward.values())
      choices=[choice for choice in reward if reward[choice]==maxValue]
      action = random.choice(choices)
      return action

    
# Abbreviation
better = betterEvaluationFunction
