import math

class Hedge:

    # Builds the Hedge algorithm with K experts and learning rate eta 
    # with initial weights set to 1 for each expert.
    #
    # Parameters:
    # - K: number of experts
    # - eta: learning rate
    def __init__(self, K, eta):
        self.K = K
        self.eta = eta
        self.weights =  [1.0 for _ in range(K)]

    # Returns the normalized weights associated to each expert at the current round.
    def get_probabilities(self):
        return self.weights
    
    def update_eta(self, eta):
        self.eta = eta

    # Update the weights of the experts based on the losses observed at the current round.
    #
    # Parameters:
    # - losses: array of losses for each expert at the current round. It must have the same length as the number of experts K.
    def update(self, losses):
        numerator = [self.weights[i] * math.exp(-self.eta * losses[i]) for i in range(self.K)]
        denominator = sum(numerator)
        self.weights = [numerator[i] / denominator for i in range(self.K)]


class AdaHedge:
    # Builds the AdaHedge algorithm with K experts and initial learning rate phi.
    #
    # Parameters:
    # - K: number of experts
    # - phi: initial learning rate for AdaHedge
    def __init__(self, K, phi=2.0):
        self.K = K
        self.phi = phi
        self.b = 0.0
        self.Delta = 0.0
        self.w = [1.0 / K for _ in range(K)]
        self.eta = 1.0
        self.b = (1.0 / (math.e - 1.0) + 1.0 / self.eta) * math.log(self.K)
        
    # Returns the normalized weights associated to each expert at the current round.
    def get_probabilities(self):
        return self.w
    
    def get_eta(self):
        return self.eta

    # Update the weights of the experts based on the losses observed at the current round and the mixability gap.
    #
    # Parameters:
    # - losses: array of losses for each expert at the current round. It must have the same length as the number of experts K.
    def update(self, losses):
        # Checking if mixability gap exceeds the budget.
        if self.Delta >= self.b:
            self.eta = self.eta / self.phi
            self.b = (1.0 / (math.e - 1.0) + 1.0 / self.eta) * math.log(self.K)
            self.Delta = 0.0
            self.w = [1.0 / self.K for _ in range(self.K)]

        # Expected losses and mixability gap update
        expected_loss = sum(self.w[i] * losses[i] for i in range(self.K))
        mix_term = (1.0 / self.eta) * math.log(sum(self.w[i] * math.exp(-self.eta * losses[i]) for i in range(self.K)))
        self.Delta += expected_loss + mix_term

        # (Normalized) weights update
        numerator = [self.w[i] * math.exp(-self.eta * losses[i]) for i in range(self.K)]
        denominator = sum(numerator)
        self.w = [numerator[i] / denominator for i in range(self.K)]