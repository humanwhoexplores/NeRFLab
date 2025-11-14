# fl_utils.py
import torch
from DecentNerfs import Client, ClientConfig, TinyNeRF

def init_global_state(device="cpu"):
    """
    Create a fresh TinyNeRF model and return its parameter tensors.
    Flower uses this as the initial global model state.
    """
    model = TinyNeRF().to(device)
    return [p.detach().cpu().clone() for p in model.parameters()]


def local_train_round(
    scene_dir: str,
    client_id: int,
    server_state,
    iters: int = 40,
    batch: int = 256,
    device: str = "cpu",
):
    """
    Run ONE round of local training on the given scene (Trevi or NotreDame).
    Uses your existing Client + ClientConfig code from DecentNerfs.py.
    Returns the same dict as client.local_train().
    """
    cfg = ClientConfig(
        id=client_id,
        scene_dir=scene_dir,
        device=device,
        iters_per_round=iters,
        batch_size=batch,
    )

    client = Client(
        cfg=cfg,
        global_mlp=TinyNeRF(),
        personal_mlp=TinyNeRF(),
    )

    return client.local_train(server_state)
