def to_numpy(tensor):
    if tensor.is_cuda:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.detach().numpy()