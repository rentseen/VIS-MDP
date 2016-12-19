def getQValue(mdp, state, a, V):
    """
    Q值，即在s下采取动作a算得的Vs
    """
    nextStates = mdp.getT(state, a)
    q = 0.0
    for (nextState, p) in nextStates:
        q += p * (mdp.getReward(state) + mdp.discount * V[nextState])

    return q

def runVI(mdp):
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
            newV[state] = max([getQValue(mdp, state, a, V) 
                               for a in mdp.getActions(state)])
            delta = max((delta, abs(v - newV[state])))
        V = newV
    
    return V

def getPolicy(mdp, V):
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
            qvalue = getQValue(mdp, state, a, V)
            if qvalue > bestValue:
                bestValue = qvalue
                bestAction = a
        policy[state] = bestAction

    return policy

