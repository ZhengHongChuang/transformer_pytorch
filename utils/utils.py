import torch


def load_checkpoint(model_path,device,is_eval=True):
    if is_eval:
        model = torch.load(model_path+'/best_model.pt')
        model.eval()
        return model.to(device)
    model = torch.load(model_path + '/last_model.pt')
    global_step = torch.load(model_path + '/global_step.pt')
    return model.to(device=device), global_step

