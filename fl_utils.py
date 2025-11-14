# fl_utils.py
from DecentNerfs import Client, ClientConfig, TinyNeRF, assign_from_flat

def init_global_state(device="cpu"):
    """
    Create a fresh TinyNeRF and return its parameters as a list of tensors.
    Suitable as an initial 'server_state' in federated settings.
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
    Helper wrapper to run ONE local training round for a single client
    using your existing architecture.
    Returns the same dict produced by Client.local_train().
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
