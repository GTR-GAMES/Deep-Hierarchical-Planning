import wandb
import torch
from pathlib import Path
import json

class Logger:
    def __init__(self, enabled=True, project="deep-hierarchical-planning", config=None, name=None):
        self.enabled = enabled
        self.project = project
        self.name = name
        self.config = config

        if enabled:
            wandb.init(project=self.project, config=self.config, name=self.name)

    def log(self, data, step=None):
        if self.enabled:
            wandb.log(data, step=step)

    def watch(self, model, log_freq=100):
        if self.enabled:
            wandb.watch(model, log='all', log_freq=log_freq)

    def log_video(self, video_tensor, caption="Evaluation", step=None, fps=30):
        if self.enabled:
            self.log({f"videos/{caption}": wandb.Video(video_tensor, fps=fps, caption=caption)}, step=step)

    def save_model(self, obj, path, artifact_name="trained-model", obj_type='auto'):
        """Save any object (model, optimizer, etc.) and log it as a W&B artifact"""
        if not self.enabled:
            return
        
        # Determine object type
        if obj_type == 'auto':
            if isinstance(obj, torch.nn.Module):
                obj_type = 'model'
            elif isinstance(obj, torch.optim.Optimizer):
                obj_type = 'optimizer'
            elif isinstance(obj, (str, Path)):
                obj_type = 'file'
            else:
                obj_type = 'checkpoint'
        
        # Handle file paths
        if isinstance(obj, (str, Path)):
            model_path = Path(obj)
            if not model_path.exists():
                raise FileNotFoundError(f"File {model_path} does not exist")
        else:
            # Save the object's state dict
            model_path = Path(path)
            if isinstance(obj, torch.nn.Module):
                torch.save(obj.state_dict(), model_path)
            elif isinstance(obj, torch.optim.Optimizer):
                torch.save(obj.state_dict(), model_path)
            else:
                torch.save(obj, model_path)
        
        # Create and log artifact with appropriate type
        artifact = wandb.Artifact(artifact_name, type=obj_type)
        artifact.add_file(str(model_path))
        wandb.log_artifact(artifact)
        
        print(f"Logged {obj_type} '{artifact_name}' to W&B")

    def finish(self):
        if self.enabled:
            wandb.finish()