import logging
import torch

from typing import Optional
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader

from mood.baselines import construct_kernel, get_baseline_model, MOOD_BASELINES
from mood.constants import NUM_EPOCHS
from mood.dataset import SimpleMolecularDataset, DAMolecularDataset, domain_based_collate
from mood.model import MOOD_DA_DG_ALGORITHMS, is_domain_generalization, is_domain_adaptation
from mood.model.base import Ensemble
from mood.model.nn import get_simple_mlp


def train_baseline_model(
    X,
    y,
    algorithm: str,
    is_regression: bool,
    params: Optional[dict] = None,
    seed: Optional[int] = None,
    for_uncertainty_estimation: bool = False,
    ensemble_size: int = 10,
    calibrate: bool = False,
    n_jobs: int = -1,
):
    if params is None:
        params = {}
    if seed is not None:
        params["random_state"] = seed

    if algorithm == "RF" or (algorithm == "GP" and not is_regression):
        params["n_jobs"] = n_jobs

    if algorithm == "RF" and not is_regression:
        params["class_weight"] = "balanced"
    if algorithm == "GP":
        params["kernel"], params = construct_kernel(is_regression, params)

    model = get_baseline_model(
        name=algorithm,
        is_regression=is_regression,
        params=params,
        for_uncertainty_estimation=for_uncertainty_estimation,
        ensemble_size=ensemble_size,
        calibrate=calibrate,
    )
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
    batch_size = params["batch_size"]

    # NOTE: Since the datasets are all very small,
    #   setting up and syncing the threads takes longer than
    #   what we gain by using the threads
    no_workers = 0

    models = []
    for i in range(ensemble_size):
        base = get_simple_mlp(len(train_dataset.X[0]), width, depth, out_size=None)
        head = get_simple_mlp(
            input_size=width * 2 if algorithm == "MTL" else width, is_regression=is_regression
        )

        model = MOOD_DA_DG_ALGORITHMS[algorithm](
            base_network=base,
            prediction_head=head,
            loss_fn=torch.nn.MSELoss() if is_regression else torch.nn.BCELoss(),
            **params,
        )

        if is_domain_adaptation(model):
            train_dataset_da = DAMolecularDataset(source_dataset=train_dataset, target_dataset=test_dataset)
            train_dataloader = DataLoader(
                train_dataset_da, batch_size=batch_size, shuffle=True, num_workers=no_workers
            )
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=no_workers)
        elif is_domain_generalization(model):
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=domain_based_collate,
                num_workers=no_workers,
            )
            val_dataloader = DataLoader(
                val_dataset, batch_size=batch_size, collate_fn=domain_based_collate, num_workers=no_workers
            )
        else:
            train_dataloader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=no_workers
            )
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=no_workers)

        # NOTE: For smaller dataset, moving data between and CPU and GPU will be the bottleneck
        use_gpu = torch.cuda.is_available() and len(train_dataset) > 2500

        callbacks = [EarlyStopping("val_loss", patience=10, mode="min")]
        trainer = Trainer(
            max_epochs=NUM_EPOCHS,
            deterministic="warn",
            callbacks=callbacks,
            enable_model_summary=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
            logger=False,
            accelerator="gpu" if use_gpu else None,
            devices=1 if use_gpu else None,
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
    calibrate: bool = False,
    ensemble_size: int = 5,
):
    # NOTE: The order here matters since there are two MLP implementations
    #  In this case, we want to use the torch implementation.

    if algorithm in MOOD_DA_DG_ALGORITHMS:
        if calibrate:
            raise NotImplementedError("We only support calibration for scikit-learn models")

        return train_torch_model(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            algorithm=algorithm,
            is_regression=is_regression,
            params=params,
            seed=seed,
            ensemble_size=ensemble_size,
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
            ensemble_size=ensemble_size,
            calibrate=calibrate,
        )

    else:
        raise NotImplementedError(f"{algorithm} is not supported")
