import pickle

import numpy as np
import torch

flame_path = "../assets/generic_model.pkl"  # flame 2020 model

with open(flame_path, "rb") as f:
    flame_ckpt = pickle.load(f, encoding="latin1")

print(flame_ckpt.keys())

tensor_version = {}
tensor_version["f"] = torch.tensor(flame_ckpt["f"], dtype=torch.long)
tensor_version["v_template"] = torch.tensor(flame_ckpt["v_template"], dtype=torch.float32)
tensor_version["shapedirs"] = torch.tensor(flame_ckpt["shapedirs"].r, dtype=torch.float32)
tensor_version["posedirs"] = torch.tensor(flame_ckpt["posedirs"], dtype=torch.float32)
tensor_version["J_regressor"] = torch.tensor(flame_ckpt["J_regressor"].todense(), dtype=torch.float32)
tensor_version["kintree_table"] = torch.tensor(flame_ckpt["kintree_table"], dtype=torch.long)
tensor_version["weights"] = torch.tensor(flame_ckpt["weights"], dtype=torch.float32)

torch.save(tensor_version, "../assets/flame_2020.pt")
