import argparse
import os
import tempfile
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from models import GNEpropGIN
from data import MolDatasetOD


class GNEprop(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.hparams = kwargs
        self.task = kwargs.get("task", "binary_classification")
        self.mpn_layers = GNEpropGIN(
            in_channels=kwargs.get("node_feat_size", 133),
            edge_dim=kwargs.get("edge_feat_size", 12),
            hidden_channels=kwargs.get("hidden_size", 500),
            ffn_hidden_channels=kwargs.get("ffn_hidden_size"),
            num_layers=kwargs.get("depth", 5),
            out_channels=kwargs.get("out_channels", 1),
            dropout=kwargs.get("dropout", 0.0),
            num_readout_layers=kwargs.get("num_readout_layers", 1),
            mol_features_size=kwargs.get("mol_features_size", 0),
            aggr=kwargs.get("aggr", "mean"),
            jk=kwargs.get("jk", "cat"),
        )

    @staticmethod
    def load_from_checkpoint(path):
        ckpt = torch.load(path, map_location="cpu")
        model = GNEprop(**ckpt["hyper_parameters"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model

    def forward(self, batch):
        logits = self.mpn_layers(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            getattr(batch, "mol_features", None),
            batch.batch
        )
        if self.task == "binary_classification":
            return torch.sigmoid(logits)
        elif self.task == "multiclass":
            return torch.softmax(logits, dim=1)
        return logits


def load_data(path):
    df = pd.read_csv(path)
    if df.shape[1] != 1:
        raise ValueError(f"Expected CSV with exactly one column, got {df.shape[1]} columns.")
    original_header = df.columns[0]
    smiles_series = df[original_header].astype(str)
    tmp_df = pd.DataFrame({"SMILES": smiles_series})
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
        tmp_df.to_csv(tmp, index=False)
    dataset = MolDatasetOD.load_csv_dataset(tmp_path)
    os.remove(tmp_path)
    loader = DataLoader(dataset, batch_size=128)
    return loader, smiles_series.tolist()


def predict_single_checkpoint(model, loader):
    preds = []
    with torch.no_grad():
        for batch in loader:
            p = model(batch).cpu().numpy().flatten()
            preds.append(p)
    return np.concatenate(preds)


def predict_ensemble(checkpoints, loader, aggr):
    preds_list = []
    for ckpt in checkpoints:
        print("Predicting:", ckpt)
        model = GNEprop.load_from_checkpoint(ckpt)
        preds = predict_single_checkpoint(model, loader)
        preds_list.append(preds)
    if len(preds_list) == 0:
        raise ValueError("No predictions collected: did not find any valid checkpoints.")
    all_preds = np.stack(preds_list)
    if aggr == "mean":
        return all_preds.mean(axis=0)
    else:
        return all_preds.max(axis=0)


def get_checkpoints(path):
    if os.path.isfile(path) and path.endswith(".ckpt"):
        return [path]
    ckpts = []
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith(".ckpt"):
                    ckpts.append(os.path.join(root, f))
    ckpts.sort()
    if not ckpts:
        raise ValueError(f"No .ckpt files found in '{path}' (searched recursively).")
    return ckpts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--target_name")
    parser.add_argument("--supervised_pretrain_path", required=True)
    parser.add_argument("--split_type")
    parser.add_argument("--lr")
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--num_readout_layers", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--lr_strategy")
    parser.add_argument("--aggr", default="mean")
    parser.add_argument("--max_epochs")
    parser.add_argument("--metric")
    parser.add_argument("--num_workers")
    parser.add_argument("--log_directory")
    parser.add_argument("--parallel_folds")
    parser.add_argument("--mp_to_freeze")
    parser.add_argument("--freeze_ab_embeddings", action="store_true")
    parser.add_argument("--freeze_batchnorm", action="store_true")
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    loader, smiles = load_data(args.dataset_path)
    checkpoints = get_checkpoints(args.supervised_pretrain_path)
    preds = predict_ensemble(checkpoints, loader, args.aggr)

    if len(preds) != len(smiles):
        raise ValueError(
            f"Length mismatch: {len(preds)} predictions vs {len(smiles)} input rows."
        )

    df = pd.DataFrame({"SMILES": smiles, "prediction": preds})
    df.to_csv(args.output_path, index=False)
    print(df.head())
    print("Saved predictions.csv")


if __name__ == "__main__":
    main()