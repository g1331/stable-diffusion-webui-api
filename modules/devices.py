import torch

# has_mps is only available in nightly pytorch (for now), `getattr` for compatibility
from modules import errors

has_mps = getattr(torch, 'has_mps', False)

cpu = torch.device("cpu")


def get_optimal_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("mps") if has_mps else cpu


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def enable_tf32():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


errors.run(enable_tf32, "Enabling TF32")


device = get_optimal_device()
device_codeformer = cpu if has_mps else device


def randn(seed, shape):
    # Pytorch currently doesn't handle setting randomness correctly when the metal backend is used.
    if device.type == 'mps':
        generator = torch.Generator(device=cpu)
        generator.manual_seed(seed)
        return torch.randn(shape, generator=generator, device=cpu).to(device)
    torch.manual_seed(seed)
    return torch.randn(shape, device=device)


def randn_without_seed(shape):
    # Pytorch currently doesn't handle setting randomness correctly when the metal backend is used.
    if device.type == 'mps':
        generator = torch.Generator(device=cpu)
        return torch.randn(shape, generator=generator, device=cpu).to(device)
    return torch.randn(shape, device=device)

