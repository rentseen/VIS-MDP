# coding=utf-8

from GridWorld import *
import json

class ComputeEngine:
    def __init__(self, costString):
        costString=costString.replace("'","")
        self.gridworld = GridWorld(3, 3, None , 0.1, 0.3)
        cost=self.stringToDic(costString)
        self.gridworld.setCost(cost)
        self.v = self.runVI(self.gridworld)
        self.policy = self.getPolicy(self.gridworld, self.v)


    def computeDepth2(self):
        """
        计算depth2的Json
        """
        gridworlddepth2 = ""
        gridworlddepth2 += json.dumps(self.outputToJSON(self.gridworld, self.gridworld.getInitialState(), None, 1.0,
                                                        self.v, self.policy, 2, True, 0))
        return gridworlddepth2


    def computeTable(self):
        """
        计算table信息
        """
        return self.outputTable(self.gridworld, self.v)
        pass

    def computePolicyTreeDepth5(self):
        """
        计算dpolicydepth5的Json
        """
        gridworldpolicytreedepth5 = ""
        gridworldpolicytreedepth5 += json.dumps(self.outputPolicyTreeToJSON(self.gridworld, self.gridworld.getInitialState(),
                                                                            None, False, 1.0, self.v, self.policy, 5, True, 0),
                                                indent=4)
        return gridworldpolicytreedepth5

    def stringToDic(self,s):
        dic={}
        tmp=s.split(',')
        states=self.gridworld.getStates()
        for s in states:
            (x,y)=s
            dic[s]=float(tmp[x+y*3])
        return dic

    # state, bestAction, value
    def outputStateValues(self, mdp, v):
        f = open('../data/gridworldstatevalues', 'w')
        f.write('state,value\n')
        for state in mdp.getStates():
            bestAction = None
            bestActionQValue = -1290310231
            for a in mdp.getActions(state):
                qvalue = self.getQValue(mdp, state, a, v)
                if qvalue > bestActionQValue:
                    bestAction = a
            f.write('%s,%s,%f\n' % (mdp.getStateString(state), bestAction, v[state]))

    # state, action, qvalue
    def outputQValues(self, mdp, v):
        f = open('../data/gridworldqvalues', 'w')
        f.write('state,action,qvalue\n')
        for state in mdp.getStates():
            for action in mdp.getActions(state):
                qvalue = 0.0
                for (s, p) in mdp.getT(state, action):
                    qvalue += p * v[s]
                f.write('%s,%s,%f\n' % (mdp.getStateString(state), str(action), qvalue))

    # state1, action, qvalue,  state2, bestAction, value, poo, probability
    def outputTable(self, mdp, v):
        table = ""
        table += 'state1,action,qvalue,qreward,state2,bestAction,value,reward,poo,incomingProbability\n'
        for state in mdp.getStates():
            for action in mdp.getActions(state):
                qvalue = 0.0
                qreward = 0.0
                for (s, p) in mdp.getT(state, action):
                    qvalue += p * v[s]
                    qreward += p * mdp.getReward(s)
                lastPointOfOrigin = -0.5
                for (s, p) in mdp.getT(state, action):
                    bestAction = None
                    bestActionQValue = -1290310231
                    for a in mdp.getActions(s):
                        q = self.getQValue(mdp, s, a, v)
                        if q > bestActionQValue:
                            bestActionQValue = q
                            bestAction = a
                    nextPointOfOrigin = lastPointOfOrigin + p / 2
                    lastPointOfOrigin += p
                    table += '%s,%s,%f,%f,%s,%s,%f,%f,%f,%f\n' % (mdp.getStateString(state), str(action), qvalue, qreward,
                                                             mdp.getStateString(s), bestAction, v[s], mdp.getReward(s),
                                                             nextPointOfOrigin, p)
        return table

    def outputPolicyTreeToJSON(self, mdp, state, action, isBestAction, prob, v,
                               policy, depth, isState, pointOfOrigin):
        json = {}
        if isState:
            json['name'] = mdp.getStateString(state)
            json['value'] = "%f" % v[state]
            json['incomingProbability'] = "%f" % prob
            if depth > 0:
                actions = []
                bestAction = None
                bestActionQValue = -1290310231
                for a in mdp.getActions(state):
                    qvalue = self.getQValue(mdp, state, a, v)
                    if qvalue > bestActionQValue:
                        bestAction = a
                        bestActionQValue = qvalue
                for a in mdp.getActions(state):
                    if a == bestAction:
                        actions.append(self.outputPolicyTreeToJSON(mdp, state, a, True, 0,
                                                              v, policy, depth, False, 0))
                    else:
                        actions.append(self.outputPolicyTreeToJSON(mdp, state, a, False, 0,
                                                              v, policy, depth, False, 0))

                json['policy'] = bestAction
                json['children'] = actions
            json['type'] = "state"
            json['poo'] = "%f" % pointOfOrigin
        else:
            json['name'] = str(action)
            json['value'] = "0"

            states = []
            qvalue = 0.0

            lastPointOfOrigin = -0.5

            if isBestAction:
                for (s, p) in mdp.getT(state, action):
                    # compute necessary translation for links so they all line up nicely
                    nextPointOfOrigin = lastPointOfOrigin + p / 2
                    lastPointOfOrigin += p

                    states.append(self.outputPolicyTreeToJSON(mdp, s, None, False, p,
                                                         v, policy, depth - 1, True,
                                                         nextPointOfOrigin))
                    qvalue += p * v[s]
            else:
                for (s, p) in mdp.getT(state, action):
                    qvalue += p * v[s]

            json['qvalue'] = qvalue

            json['children'] = states
            json['type'] = "action"

        return json

    def outputToJSON(self, mdp, state, action, prob, v,
                     policy, depth, isState, pointOfOrigin):
        """

        :param mdp: GridWorld
        :param state: 初始state
        :param action:
        :param prob: 初始state的incoming prob
        :param v: mdp的state value
        :param policy: 最优策略
        :param depth: 深度
        :param isState:
        :param pointOfOrigin:
        :return:
        """
        json = {}
        if isState:
            json['name'] = mdp.getStateString(state)
            json['value'] = "%f" % v[state]
            json['incomingProbability'] = "%f" % prob
            # if depth > 0:
            actions = []
            bestAction = None
            bestActionQValue = -1290310231
            for a in mdp.getActions(state):
                qvalue = self.getQValue(mdp, state, a, v)
                if qvalue > bestActionQValue:
                    bestAction = a
                    bestActionQValue = qvalue
            if depth > 0:
                for a in mdp.getActions(state):
                    actions.append(self.outputToJSON(mdp, state, a, 0,
                                                v, policy, depth, False, 0))
                json['children'] = actions
            json['policy'] = bestAction
            json['type'] = "state"
            json['poo'] = "%f" % pointOfOrigin
        else:
            json['name'] = str(action)
            json['value'] = "0"

            states = []
            qvalue = 0.0

            lastPointOfOrigin = -0.5

            for (s, p) in mdp.getT(state, action):
                # compute necessary translation for links so they all line up nicely
                nextPointOfOrigin = lastPointOfOrigin + p / 2
                lastPointOfOrigin += p

                states.append(self.outputToJSON(mdp, s, None, p,
                                           v, policy, depth - 1, True,
                                           nextPointOfOrigin))
                qvalue += p * v[s]

            json['qvalue'] = qvalue

            json['children'] = states
            json['type'] = "action"

        return json

    def getQValue(self, mdp, state, a, V):
        """
        Q值，即在s下采取动作a算得的Vs
        """
        nextStates = mdp.getT(state, a)
        q = 0.0
        for (nextState, p) in nextStates:
            q += p * (mdp.getReward(state) + mdp.discount * V[nextState])

        return q

    def runVI(self, mdp):
        """
        值迭代
        """
        states = mdp.getStates()
        V = {}

        for state in states:
            V[state] = 0.0

        delta = 10000.0
        epsilon = 0.01

        while delta > epsilon:
            delta = 0.0
            newV = {}
            for state in states:
                v = V[state]
                newV[state] = max([self.getQValue(mdp, state, a, V)
                                   for a in mdp.getActions(state)])
                delta = max((delta, abs(v - newV[state])))
            V = newV

        return V

    def getPolicy(self, mdp, V):
        """
        获取最优policy
        :return: 字典
        """
        states = mdp.getStates()
        policy = {}
        for state in states:
            bestAction = None
            bestValue = -102391231
            for a in mdp.getActions(state):
                qvalue = self.getQValue(mdp, state, a, V)
                if qvalue > bestValue:
                    bestValue = qvalue
                    bestAction = a
            policy[state] = bestAction

        return policy

