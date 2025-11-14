# flower_client.py
import argparse
import flwr as fl
import numpy as np
from fl_utils import init_global_state, local_train_round

class NeRFClient(fl.client.NumPyClient):
    def __init__(self, dataset, cid, device="cpu"):
        self.dataset = dataset
        self.cid = cid
        self.device = device

    # Load initial weights from our TinyNeRF
    def get_parameters(self, config):
        params = init_global_state(device=self.device)
        flat = np.concatenate([p.numpy().reshape(-1) for p in params])
        return flat

    # Flower gives us global parameters -> we run ONE local round
    def fit(self, parameters, config):
        # Convert incoming flat array back to tensors
        server_state = []
        idx = 0
        for p in init_global_state():
            n = p.numel()
            chunk = parameters[idx: idx + n]
            server_state.append(torch.tensor(chunk).view_as(p))
            idx += n

        # run local round
        pkt = local_train_round(
            scene_dir=self.dataset,
            client_id=self.cid,
            server_state=server_state,
            iters=40,
            batch=256,
            device=self.device,
        )

        # return updated global weights to Flower
        updated = pkt["flat_global"].numpy()
        return updated, len(updated), {}

    def evaluate(self, parameters, config):
        return 0.0, len(parameters), {}

def start_client():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--cid", type=int, required=True)
    parser.add_argument("--server", type=str, default="localhost:8080")
    args = parser.parse_args()

    fl.client.start_numpy_client(
        server_address=args.server,
        client=NeRFClient(dataset=args.dataset, cid=args.cid),
    )

if __name__ == "__main__":
    start_client()
