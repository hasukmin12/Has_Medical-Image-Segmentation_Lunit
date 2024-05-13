import torch
seed = 12312
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(config["SEEDS"])
# random.seed(config["SEEDS"])

rst = torch.rand(5)
torch.manual_seed(seed)
rst2 = torch.rand(5)
torch.manual_seed(seed)
rst3 = torch.rand(5)