import os
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Adjust these imports to match your project structure
# from neuralforge.models.fno1d import FNO1D
from neuralforge.training.losses import LpLoss
from neuralforge.training.trainer import Trainer

# --- DUMMY FIXTURES FOR ISOLATED TESTING ---


def make_dummy_loader(n_samples=20, n=64, da=1, du=1, batch_size=4):
    """Create a small DataLoader for testing."""
    x = torch.randn(n_samples, n, da)
    y = torch.randn(n_samples, n, du)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size)


class DummyModel(nn.Module):
    """
    A tiny model to test the Trainer without depending on FNO1D.
    Takes input (batch, n, da) and outputs (batch, n, du).
    """

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # A simple linear layer that maps 'da' to 'du'
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.linear(x)


# --- THE TESTS ---


def test_trainer_instantiates_without_error():
    """1. Test that the Trainer initializes properly with all components."""
    model = DummyModel()
    loader = make_dummy_loader()
    loss_fn = LpLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=loader,
        val_loader=loader,
        device="cpu",
    )

    assert trainer is not None
    assert trainer.device == "cpu"
    assert trainer.best_val_loss == float("inf")


def test_one_epoch_of_training_runs():
    """2. Test that one full training epoch executes and returns a valid float."""
    model = DummyModel()
    loader = make_dummy_loader(n_samples=8, batch_size=4)  # Keep it small for speed
    loss_fn = LpLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    trainer = Trainer(model, loss_fn, optimizer, loader, loader, device="cpu")

    avg_loss = trainer._train_one_epoch()

    assert isinstance(avg_loss, float)
    assert not torch.isnan(torch.tensor(avg_loss)), "Loss became NaN!"
    assert avg_loss >= 0.0, "Relative L2 loss cannot be negative."


def test_training_loss_finite_and_positive():
    """3. Test multiple epochs to ensure loss remains valid (and ideally decreases)."""
    model = DummyModel()
    loader = make_dummy_loader(n_samples=8, batch_size=4)
    loss_fn = LpLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    trainer = Trainer(model, loss_fn, optimizer, loader, loader, device="cpu")

    # Run two epochs manually
    loss_epoch_1 = trainer._train_one_epoch()
    loss_epoch_2 = trainer._train_one_epoch()

    # We test for finiteness and positivity. We don't strictly assert
    # loss_epoch_2 < loss_epoch_1 because random initialization and
    # small batch sizes can occasionally cause loss spikes, which
    # would lead to a flaky unit test.
    assert loss_epoch_1 >= 0.0
    assert loss_epoch_2 >= 0.0
    assert not torch.isnan(torch.tensor(loss_epoch_1))
    assert not torch.isnan(torch.tensor(loss_epoch_2))


def test_checkpoint_saved_when_val_loss_improves():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = DummyModel()
        loader = make_dummy_loader(n_samples=8, batch_size=4)
        loss_fn = LpLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        trainer = Trainer(
            model,
            loss_fn,
            optimizer,
            loader,
            loader,
            device="cpu",
            checkpoint_path=tmp_dir,
        )

        trainer.fit(epochs=1)

        expected = os.path.join(tmp_dir, "best_model.pth")
        assert os.path.exists(expected), f"Checkpoint not saved at {expected}"


def test_model_is_moved_to_device_correctly():
    """5. Test that the model parameters are sent to the correct hardware."""
    model = DummyModel()
    loader = make_dummy_loader(n_samples=4, batch_size=2)
    loss_fn = LpLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Dynamically select GPU if available, otherwise CPU
    test_device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = Trainer(model, loss_fn, optimizer, loader, loader, device=test_device)

    # Check the device of the very first weight tensor in the model
    param_device = next(trainer.model.parameters()).device.type

    assert (
        param_device == test_device
    ), f"Expected model on {test_device}, but found it on {param_device}"
