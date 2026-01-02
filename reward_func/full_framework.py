import numpy as np
import torch
import torch.nn.functional as F

class SelfEvolvingRewards:
    
    def __init__(self, penalty_lambda=0.1, dpo_beta=0.1):
        self.penalty_lambda = penalty_lambda
        self.dpo_beta = dpo_beta

    def hsr_ee_reward(self, r_visual, r_ans, k):
        #  R_{HSR}=r_{visual}+r_{ans}-\lambda\cdot k
        reward = r_visual + r_ans - (self.penalty_lambda * k)
        return reward

    def mdp_step_reward(self, action, k, r_visual=None, r_ans=None):
        if action == "Exit":
            if r_visual is None or r_ans is None:
                raise ValueError("Action 'Exit' requires r_visual and r_ans values.")
            return r_visual + r_ans - (self.penalty_lambda * k)
        
        elif action == "Continue":
            return -self.penalty_lambda
        
        else:
            raise ValueError("Action must be 'Exit' or 'Continue'")

    def sr_dpo_loss(self, policy_chosen_logps, policy_rejected_logps, 
                    ref_chosen_logps, ref_rejected_logps):

        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps
        
        logits = self.dpo_beta * (chosen_logratios - rejected_logratios)
        
        losses = -F.logsigmoid(logits)
        
        return losses.mean()