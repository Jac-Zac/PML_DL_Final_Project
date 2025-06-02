from typing import Callable, Sequence


def load_pretrained_model(args: MLPTrainingArgs, model_path: str, device: torch.device):
    """
    Load a trained model from the given path.
    """
    model = Model(hidden_sizes=args.hidden_sizes)

    # Ensure model_path is valid and remove invalid argument
    model.load_state_dict(torch.load(model_path, weights_only=True))

    return model.to(device)  # Move model to specified device
