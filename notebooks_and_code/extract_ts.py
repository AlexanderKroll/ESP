#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pathlib

import torch

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained


def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file"  # noqa
    )

    parser.add_argument(
        "model_location",
        type=str,
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
    )
    parser.add_argument(
        "fasta_file",
        type=pathlib.Path,
        help="FASTA file on which to extract representations",
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="output directory for extracted representations",
    )

    parser.add_argument(
        "--toks_per_batch", type=int, default=4096, help="maximum batch size"
    )
    parser.add_argument(
        "--repr_layers",
        type=int,
        default=[-1],
        nargs="+",
        help="layers indices from which to extract representations (0 to num_layers, inclusive)",
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="+",
        choices=["mean", "per_tok", "bos", "contacts"],
        help="specify which representations to return",
        required=True
    )
    parser.add_argument(
        "--truncate", 
        action="store_true", 
        help="Truncate sequences longer than 1024 to match the training setup"
    )

    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return parser


def main(args):
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    model.eval()
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")

    PATH = '/gpfs/project/alkro105/ESM/models/model_ESM_binary_A100_epoch_1_new_split.pkl'
    model_dict = torch.load(PATH, map_location="cuda")
    model_dict_V2 = {k.split("model.")[-1]: v for k, v in model_dict.items()}

    for key in ["module.fc1.weight", "module.fc1.bias", "module.fc2.weight", "module.fc2.bias", "module.fc3.weight", "module.fc3.bias"]:
        del model_dict_V2[key]
    model.load_state_dict(model_dict_V2)

    dataset = FastaBatchedDataset.from_file(args.fasta_file)
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches)
    print(f"Read {args.fasta_file} with {len(dataset)} sequences")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    return_contacts = "contacts" in args.include

    assert all(
        -(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers
    )
    repr_layers = [
        (i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers
    ]

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(labels)
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available() and not args.nogpu:
                toks = toks.to(device="cuda", non_blocking=True)

            # The model is trained on truncated sequences and passing longer ones in at
            # infernce will cause an error. See https://github.com/facebookresearch/esm/issues/21
            if args.truncate:
                toks = toks[:, :1022]

            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)
            #output = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)
            #output = output["representations"][rep_layer]
            #output = output[:,0,:]



            logits = out["logits"].to(device="cpu")
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }
            if return_contacts:
                contacts = out["contacts"].to(device="cpu")

            for i, label in enumerate(labels):

                args.output_file = (
                    args.output_dir / f"{label}.pt"
                )
                args.output_file.parent.mkdir(parents=True, exist_ok=True)
                result = {"label": label}
     
                if "mean" in args.include:
                    result["cls_representations"] = {
                        layer: t[i,0,:].clone()
                        for layer, t in representations.items()
                    }

                torch.save(
                    result,
                    args.output_file,
                )


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
