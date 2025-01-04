import os
import torch

def save_checkpoint(model_info, directory) :
    """Saves a checkpoint of the network and other variables."""
    tmp_file_path = os.path.join(directory, f"checkpoint_ep{model_info['epoch']:04d}.pth")
    torch.save(model_info, tmp_file_path)
    print("Model has been saved to {}".format(tmp_file_path))