import torch
import torch.nn.functional as F

from typing import Optional


class PGDAttack:
    """
    Projected Gradient Descent (PGD) Attack for Whisper.
    """
    
    def __init__(
        self, 
        model, 
        epsilon: float = 0.01, 
        alpha: float = 0.003, 
        num_iter: int = 10, 
        attack_type: str = "untargeted",
        random_start: bool = True
    ):
        """
        Args:
            model: Whisper model instance.
            epsilon: Max perturbation magnitude (L_inf norm).
            alpha: Step size for PGD.
            num_iter: Number of optimization steps.
            attack_type: 'untargeted' or 'targeted'.
            random_start: Whether to initialize with random noise.
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.attack_type = attack_type
        self.random_start = random_start
        
        # Ensure model is in eval mode for attack
        self.model.eval()
    
    def _to_device(self, tensor):
        return tensor.to(self.model.device if hasattr(self.model, 'device') else 'cpu')

    def generate(self, audio: torch.Tensor, input_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Deep copy to avoid modifying original audio
        adv_audio = audio.clone().detach()
        
        if self.random_start:
            start_delta = torch.empty_like(adv_audio).uniform_(
                -self.epsilon, self.epsilon
            )
            adv_audio = torch.clamp(adv_audio + start_delta, -1.0, 1.0)

        for _ in range(self.num_iter):
            adv_audio.requires_grad = True
            
            with torch.enable_grad():
                # Check if model has a dedicated loss function for attacks
                if hasattr(self.model, "get_loss_for_attack"):
                    loss = self.model.get_loss_for_attack(adv_audio)
                else:
                    output = self.model(adv_audio)
                    logits = output.logits
                    # Simple untargeted: Maximize uncertainty (entropy) of first token
                    # or just minimize max logit
                    probs = F.softmax(logits[:, 0, :], dim=-1)
                    loss = -torch.max(probs, dim=-1)[0].mean()
                
            # Compute gradient
            grad = torch.autograd.grad(loss, adv_audio, retain_graph=False)[0]
            
            # Update (Sign of gradient)
            if self.attack_type == "untargeted":
                # maximize the loss (maximize uncertainty)
                # So we move in the direction of the gradient
                adv_audio = adv_audio + self.alpha * torch.sign(grad)
            elif self.attack_type == "targeted":
                # minimize the loss (minimize distance to target)
                # So we move against the gradient
                adv_audio = adv_audio - self.alpha * torch.sign(grad)

            # Project back to epsilon-ball
            adv_audio = torch.clamp(adv_audio, audio - self.epsilon, audio + self.epsilon)
            adv_audio = torch.clamp(adv_audio, -1.0, 1.0) # Ensure valid audio range
            
            # Detach for next iteration
            adv_audio = adv_audio.detach()

        return adv_audio


