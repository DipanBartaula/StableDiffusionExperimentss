import importlib.util
import os

import torch
import torch.nn.functional as F


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    mod_path = os.path.join(os.path.dirname(__file__), "train_cpvton_local.py")
    cp = load_module(mod_path, "cp_train_local")

    device = torch.device("cpu")
    b, h, w = 2, 128, 128
    person = torch.randn(b, 3, h, w, device=device)
    cloth = torch.randn(b, 3, h, w, device=device)
    gt = torch.randn(b, 3, h, w, device=device)

    gmm = cp.GMM().to(device).train()
    warped, grid, theta = gmm(person, cloth)
    assert warped.shape == cloth.shape, f"Warped shape mismatch: {warped.shape}"
    assert grid.shape[:3] == (b, h, w), f"Grid shape mismatch: {grid.shape}"
    assert theta.shape[0] == b and theta.shape[2] == 2, f"Theta shape mismatch: {theta.shape}"
    gmm_loss = F.l1_loss(warped, gt) + 0.01 * theta.pow(2).mean()
    gmm_loss.backward()
    assert gmm.features[0].weight.grad is not None, "No grad in GMM feature extractor"

    tom = cp.TOM().to(device).train()
    mask = (person - gt).abs().mean(dim=1, keepdim=True).clamp(0, 1)
    person_agnostic = person * (1 - mask)
    final, rendered, comp_mask = tom(person_agnostic, warped.detach())
    assert final.shape == gt.shape, f"Final shape mismatch: {final.shape}"
    assert rendered.shape == gt.shape, f"Rendered shape mismatch: {rendered.shape}"
    assert comp_mask.shape == (b, 1, h, w), f"Mask shape mismatch: {comp_mask.shape}"
    tom_loss = F.l1_loss(final, gt) + F.l1_loss(rendered, gt) + 0.01 * cp.tv_loss(comp_mask)
    tom_loss.backward()
    assert tom.unet.e1.net[0].weight.grad is not None, "No grad in TOM UNet"
    print("CPVTON forward/backward PASS (GMM + TOM)")


if __name__ == "__main__":
    main()
