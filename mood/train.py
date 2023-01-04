import logging
from typing import Optional

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from lightning_lite.utilities.seed import seed_everything
from torch.utils.data import DataLoader

from mood.baselines import construct_kernel, get_baseline_model, MOOD_BASELINES
from mood.dataset import SimpleMolecularDataset, DAMolecularDataset, domain_based_collate
from mood.model import MOOD_ALGORITHMS, is_domain_generalization, is_domain_adaptation
from mood.model.base import Ensemble
from mood.model.nn import get_simple_mlp
from mood.utils import Timer


BATCH_SIZE = 256
NUM_EPOCHS = 100


def train_baseline_model(
    X,
    y,
    algorithm: str,
    is_regression: bool,
    params: Optional[dict] = None,
    seed: Optional[int] = None,
    for_uncertainty_estimation: bool = False,
    ensemble_size: int = 10,
):
    if params is None:
        params = {}
    if seed is not None:
        params["random_state"] = seed

    if algorithm == "RF" and not is_regression:
        params["class_weight"] = "balanced"
    if algorithm == "GP":
        params["kernel"], params = construct_kernel(is_regression, params)

    model = get_baseline_model(algorithm, is_regression, params, for_uncertainty_estimation, ensemble_size)
    model.fit(X, y)

    return model


def train_torch_model(
    train_dataset: SimpleMolecularDataset,
    val_dataset: SimpleMolecularDataset,
    test_dataset: SimpleMolecularDataset,
    algorithm: str,
    is_regression: bool,
    params: Optional[dict] = None,
    seed: Optional[int] = None,
    ensemble_size: int = 5,
):

    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    seed_everything(seed, workers=True)

    width = params.pop("mlp_width")
    depth = params.pop("mlp_depth")

    base = get_simple_mlp(len(train_dataset.X[0]), width, depth, out_size=None)
    head = get_simple_mlp(input_size=width, is_regression=is_regression)

    models = []
    for i in range(ensemble_size):
        model = MOOD_ALGORITHMS[algorithm](
            base_network=base,
            prediction_head=head,
            loss_fn=torch.nn.MSELoss() if is_regression else torch.nn.BCELoss(),
            batch_size=BATCH_SIZE,
            **params
        )

        if is_domain_adaptation(model):
            train_dataset_da = DAMolecularDataset(source_dataset=train_dataset, target_dataset=test_dataset)
            train_dataloader = DataLoader(train_dataset_da, batch_size=BATCH_SIZE, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        elif is_domain_generalization(model):
            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=domain_based_collate)
            val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=domain_based_collate)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        callbacks = [EarlyStopping("val_loss", patience=10, mode="min")]
        trainer = Trainer(
            max_epochs=NUM_EPOCHS,
            deterministic="warn",
            callbacks=callbacks,
            enable_model_summary=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(model, train_dataloader, val_dataloader)
        models.append(model)

    return Ensemble(models, is_regression)


def train(
        train_dataset: SimpleMolecularDataset,
        val_dataset: SimpleMolecularDataset,
        test_dataset: SimpleMolecularDataset,
        algorithm: str,
        is_regression: bool,
        params: dict,
        seed: int,
):
    # NOTE: The order here matters since there are two MLP implementations
    #  In this case, we want to use the torch implementation.

    if algorithm in MOOD_ALGORITHMS:
        return train_torch_model(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            algorithm=algorithm,
            is_regression=is_regression,
            params=params,
            seed=seed,
            ensemble_size=5,
        )

    elif algorithm in MOOD_BASELINES:
        return train_baseline_model(
            X=train_dataset.X,
            y=train_dataset.y,
            algorithm=algorithm,
            is_regression=is_regression,
            params=params,
            seed=seed,
            for_uncertainty_estimation=True,
            ensemble_size=5
        )

    else:
        raise NotImplementedError(f"{algorithm} is not supported")

