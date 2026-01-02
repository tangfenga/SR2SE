import torch
import torch.nn.functional as F

class AblationConfig:
    def __init__(self, name, use_hsr_ee=True, use_ipr=True, use_sr_dpo=True):
        self.name = name
        self.use_hsr_ee = use_hsr_ee   
        self.use_ipr = use_ipr         
        self.use_sr_dpo = use_sr_dpo   

class VLM_Ablation_Agent:
    def __init__(self, config):
        self.config = config

    def inference(self, image, query):
        history = []
        
        if self.config.use_hsr_ee:
            max_dynamic_steps = 10
            for k in range(max_dynamic_steps):
                step_content = f"Reasoning step {k}"
                history.append(step_content)
                gate_score = torch.rand(1).item() 
                
                if gate_score > self.gate_threshold:
                    print(f"[{self.config.name}] HSR-EE Triggered: Early Exit at step {k+1}")
                    break
        else:
            for k in range(self.static_steps):
                step_content = f"Reasoning step {k}"
                history.append(step_content)
            print(f"[{self.config.name}] Static Graph: Forced exit at fixed step {self.static_steps}")

        final_answer = "Generated Answer"
        return history, final_answer

    def run_data_engine(self, image, query, initial_perception, ground_truth):
        is_valid = (initial_perception == ground_truth) 
        
        if is_valid:
            return None 

        if self.config.use_ipr:

            refined_perception = "Refined perception with details"

            refined_is_valid = True 
            
            if refined_is_valid:
                return {"winner": refined_perception, "loser": initial_perception}
            return None
        else:
            print(f"[{self.config.name}] IPR Disabled: Failure discarded.")
            return None

    def calculate_loss(self, batch_data, model_outputs):
        if self.config.use_sr_dpo:
            pi_w = model_outputs['log_prob_winner']
            pi_l = model_outputs['log_prob_loser']
            ref_w = model_outputs['ref_log_prob_winner']
            ref_l = model_outputs['ref_log_prob_loser']
            
            logits = self.beta * ((pi_w - ref_w) - (pi_l - ref_l))
            loss = -F.logsigmoid(logits).mean()
            print(f"[{self.config.name}] Calculating SR-DPO Loss...")
            return loss
            
        else:
            advantage = model_outputs['advantage']
            ratio = model_outputs['ratio']
            epsilon = 0.2

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantage
            loss = -torch.min(surr1, surr2).mean()
            print(f"[{self.config.name}] Calculating Binary PPO Loss...")
            return loss

experiments = [
    AblationConfig(name="Full Framework", use_hsr_ee=True, use_ipr=True, use_sr_dpo=True),
    
    AblationConfig(name="w/o HSR-EE", use_hsr_ee=False, use_ipr=True, use_sr_dpo=True),

    AblationConfig(name="w/o IPR", use_hsr_ee=True, use_ipr=False, use_sr_dpo=True),

    AblationConfig(name="w/o SR-DPO", use_hsr_ee=True, use_ipr=True, use_sr_dpo=False)
]

