"""
Quantization script for fish-speech s2-pro models using bitsandbytes.

This script provides INT4/INT8 quantization for DualARTransformer models
using bitsandbytes' quantization primitives.
"""

import datetime
import json
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import click
import torch
import torch.nn as nn
import torch.nn.functional as F

from bitsandbytes.nn import Linear4bit, Linear8bitLt
from loguru import logger
from safetensors.torch import load_file

from fish_speech.models.text2semantic.llama import (
    BaseTransformer,
    DualARTransformer,
    NaiveTransformer,
    BaseModelArgs,
    DualARModelArgs,
    NaiveModelArgs,
    _remap_fish_qwen3_omni_keys,
)
from fish_speech.models.text2semantic.inference import init_model


def get_model_type(checkpoint_path: Path) -> str:
    """Detect model type from config.json"""
    config_path = checkpoint_path / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    return config.get("model_type", "unknown")


def _load_safetensors_weights(checkpoint_path: Path) -> OrderedDict:
    """Load weights from safetensors files with key remapping.

    Supports both sharded (model.safetensors.index.json) and single (model.safetensors) formats.
    """
    index_json = checkpoint_path / "model.safetensors.index.json"
    single_st = checkpoint_path / "model.safetensors"

    weights = OrderedDict()

    if index_json.exists():
        logger.info("Loading sharded safetensors weights")
        with open(index_json) as f:
            st_index = json.load(f)
        shard_files = sorted(set(st_index["weight_map"].values()))
        for shard in shard_files:
            weights.update(load_file(str(checkpoint_path / shard), device="cpu"))
    elif single_st.exists():
        logger.info("Loading single safetensors weights")
        weights = OrderedDict(load_file(str(single_st), device="cpu"))
    else:
        raise FileNotFoundError(f"No safetensors weights found in {checkpoint_path}")

    # Apply key remapping for fish-qwen3-omni format
    weights = _remap_fish_qwen3_omni_keys(weights)
    return weights


def load_model(checkpoint_path: str, model_type: str = None) -> BaseTransformer:
    """Load a fish-speech s2-pro model from safetensors checkpoints.

    Args:
        checkpoint_path: Path to the model checkpoint directory
        model_type: Optional model type override (auto-detected from config if not provided)

    Returns:
        Loaded BaseTransformer model with weights applied
    """
    checkpoint_path = Path(checkpoint_path)

    # Auto-detect model type if not provided
    if model_type is None:
        model_type = get_model_type(checkpoint_path)

    logger.info(f"Loading model type: {model_type}")

    # Load config and create model args
    config = BaseModelArgs.from_pretrained(str(checkpoint_path))

    # Override model_type from config if needed
    if hasattr(config, "model_type"):
        model_type = config.model_type

    # Select model class based on model_type
    match model_type:
        case "naive":
            model_cls = NaiveTransformer
        case "dual_ar" | "fish_qwen3_omni":
            model_cls = DualARTransformer
        case _:
            raise ValueError(f"Unknown model type: {model_type}")

    # Initialize model (random weights)
    logger.info(f"Initializing {model_cls.__name__} with config")
    model = model_cls(config)

    # Load safetensors weights and apply to model
    weights = _load_safetensors_weights(checkpoint_path)
    err = model.load_state_dict(weights, strict=False, assign=True)
    logger.info(f"Model weights loaded - Status: {err}")

    return model


@dataclass
class QuantConfig:
    """Configuration for bitsandbytes quantization."""

    mode: str = "int4"  # "int4" or "int8"
    groupsize: int = 128
    compute_dtype: torch.dtype = torch.bfloat16


def load_bnb_quantized_state_dict(
    model: nn.Module,
    weights: OrderedDict,
    device: str | torch.device = "cuda",
):
    """Load a bitsandbytes-packed checkpoint into a model with Linear4bit layers."""
    from bitsandbytes.nn import Linear4bit, Linear8bitLt
    from bitsandbytes.nn.modules import Params4bit

    consumed_prefixes: list[str] = []

    for name, module in model.named_modules():
        if isinstance(module, Linear4bit):
            prefix = f"{name}."
            weight_key = f"{prefix}weight"
            if weight_key not in weights:
                logger.warning(f"Missing quantized weights for {name}")
                continue

            qs_dict = {
                k[len(f"{prefix}weight.") :]: v
                for k, v in weights.items()
                if k.startswith(f"{prefix}weight.")
            }
            module.weight = Params4bit.from_prequantized(
                weights[weight_key],
                qs_dict,
                device=device,
                module=module,
            )
            if f"{prefix}bias" in weights and module.bias is not None:
                module.bias = nn.Parameter(weights[f"{prefix}bias"].to(device=device))
            consumed_prefixes.append(prefix)
        elif isinstance(module, Linear8bitLt):
            raise NotImplementedError(
                "Loading bitsandbytes int8 checkpoints is not implemented yet."
            )

    remaining = OrderedDict()
    for key, value in weights.items():
        if any(key.startswith(prefix) for prefix in consumed_prefixes):
            continue
        remaining[key] = value

    return model.load_state_dict(remaining, strict=False, assign=True)


def _should_skip_bnb_linear(name: str) -> bool:
    """Keep embedding lookups in full precision."""
    return "embeddings" in name or "codebook_embeddings" in name


def replace_linear_with_bitsandbytes(
    model: nn.Module,
    mode: str = "int4",
    groupsize: int = 128,
    device: str | torch.device | None = None,
    copy_fp_weights: bool = True,
) -> nn.Module:
    """Replace nn.Linear layers with bitsandbytes quantized versions.

    Args:
        model: The model to quantize
        mode: Quantization mode - "int4" or "int8"
        groupsize: Quantization groupsize (default: 128)
        device: Device used when constructing quantized layers (e.g. "cuda")
        copy_fp_weights: Copy existing fp32 weights into new layers before save/load

    Returns:
        The model with quantized linear layers

    Notes:
        - INT4: Uses bitsandbytes Linear4bit (NF4); packed weights load after replace
        - INT8: Uses bitsandbytes Linear8bitLt
        - Embedding layers are kept in full precision
    """
    linear_layers: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not _should_skip_bnb_linear(name):
            linear_layers.append((name, module))

    replaced_count = 0
    skipped_count = 0

    for full_name, linear_layer in linear_layers:
        parent_name, _, child_name = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model

        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        has_bias = linear_layer.bias is not None

        try:
            if mode == "int4":
                new_layer = Linear4bit(
                    in_features,
                    out_features,
                    bias=has_bias,
                    compute_dtype=torch.bfloat16,
                    compress_statistics=groupsize == 128,
                    quant_type="nf4",
                    device=device,
                )
            elif mode == "int8":
                new_layer = Linear8bitLt(
                    in_features,
                    out_features,
                    bias=has_bias,
                    has_fp16_weights=False,
                    device=device,
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

            if copy_fp_weights:
                new_layer.load_state_dict(linear_layer.state_dict())
            setattr(parent, child_name, new_layer)
            replaced_count += 1
            logger.debug(
                f"Replaced {full_name}: {in_features}x{out_features} ({mode})"
            )

        except Exception as e:
            logger.warning(f"Failed to quantize layer {full_name}: {e}")
            skipped_count += 1

    remaining = [
        name
        for name, module in model.named_modules()
        if type(module) is nn.Linear and not _should_skip_bnb_linear(name)
    ]
    if remaining:
        raise RuntimeError(
            "bitsandbytes layer replacement incomplete; still have nn.Linear at: "
            + ", ".join(remaining[:8])
            + (" ..." if len(remaining) > 8 else "")
        )

    logger.info(
        f"Quantization complete: {replaced_count} layers quantized, "
        f"{skipped_count} skipped"
    )
    return model


@click.command()
@click.option(
    "--checkpoint-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the model checkpoint directory",
)
@click.option(
    "--mode",
    type=click.Choice(["int4", "int8"]),
    default="int4",
    help="Quantization mode (int4 or int8)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Output directory for quantized model (auto-generated if not specified)",
)
@click.option(
    "--groupsize",
    type=int,
    default=128,
    help="Quantization groupsize for INT4 mode",
)
def quantize(checkpoint_path, mode, output_dir, groupsize):
    """
    Quantize a fish-speech s2-pro model using bitsandbytes.

    Examples:

        Quantize to INT4:

            python tools/llama/quantize_s2.py --checkpoint-path /path/to/model --mode int4

        Quantize to INT8:

            python tools/llama/quantize_s2.py --checkpoint-path /path/to/model --mode int8 --groupsize 64
    """
    # Section 1 - Model Loading
    logger.info(f"Loading model from {checkpoint_path}")

    # Load model config and detect model type
    config_path = Path(checkpoint_path) / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    model_type = config.get("model_type", "unknown")
    logger.info(f"Detected model type: {model_type}")

    # Load model based on type
    model = load_model(checkpoint_path, model_type)
    logger.info("Model loaded successfully")

    # Section 2 - Quantization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning(
            "CUDA is not available. bitsandbytes 4/8-bit layers require a GPU to "
            "quantize weights; saving unquantized weights instead."
        )
    else:
        logger.info(f"Using device: {device}")

    logger.info(f"Applying {mode} quantization with groupsize={groupsize}")
    model = replace_linear_with_bitsandbytes(model, mode, groupsize)
    model = model.to(device)

    # Section 3 - Main Execution
    logger.info("=" * 60)
    logger.info("Starting quantization pipeline")
    logger.info("=" * 60)

    # Auto-generate output directory if not specified
    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%m%S")
        mode_str = f"int{groupsize}" if mode == "int4" else "int8"
        output_dir = f"checkpoints/fs-s2-{mode_str}-g{groupsize}-{timestamp}"
        logger.info(f"Auto-generated output directory: {output_dir}")

    output_dir = Path(output_dir)

    # Create output directory by copying original checkpoint
    logger.info(f"Creating output directory: {output_dir}")
    try:
        shutil.copytree(
            Path(checkpoint_path),
            output_dir,
            dirs_exist_ok=False,
        )
        logger.info("Checkpoint copied successfully")
    except FileExistsError:
        logger.error(f"Output directory already exists: {output_dir}")
        raise click.ClickException(
            f"Output directory '{output_dir}' already exists. "
            "Please specify a different --output-dir or remove the existing directory."
        )

    # Remove existing model.safetensors files (we'll save quantized weights separately)
    index_json = output_dir / "model.safetensors.index.json"
    single_st = output_dir / "model.safetensors"

    if index_json.exists():
        logger.info("Removing existing sharded safetensors weights")
        with open(index_json) as f:
            st_index = json.load(f)
        for shard in sorted(set(st_index.get("weight_map", {}).values())):
            shard_path = output_dir / shard
            if shard_path.exists():
                shard_path.unlink()
        index_json.unlink()

    if single_st.exists():
        logger.info("Removing existing safetensors file")
        single_st.unlink()

    # Save quantized model using safetensors format
    logger.info("Extracting state dict from quantized model")
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    logger.info(f"State dict contains {len(state_dict)} tensors")

    # Define paths
    safetensors_path = output_dir / "model.safetensors"
    pth_path = output_dir / "model.pth"
    logger.info(f"Saving quantized weights to {safetensors_path}")

    try:
        from safetensors.torch import save_file

        save_file(state_dict, str(safetensors_path))
        logger.info("Quantized model saved successfully")
    except Exception as e:
        logger.error(f"Failed to save safetensors: {e}")
        # Fallback to torch.save if safetensors fails
        logger.warning(f"Falling back to torch.save: {pth_path}")
        torch.save(state_dict, str(pth_path))

    # Generate quantization report
    total_params = sum(p.numel() for p in model.parameters())
    quantized_modules = [
        m for m in model.modules() if isinstance(m, (Linear4bit, Linear8bitLt))
    ]
    quantized_params = sum(
        sum(p.numel() for p in m.parameters()) for m in quantized_modules
    )

    report = {
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(output_dir),
        "mode": mode,
        "groupsize": groupsize,
        "device": str(device),
        "total_parameters": total_params,
        "quantized_parameters": quantized_params,
        "quantized_layers": len(quantized_modules),
        "original_checkpoint": str(Path(checkpoint_path).resolve()),
        "quantized_weights_file": str(
            safetensors_path if safetensors_path.exists() else pth_path
        ),
    }

    report_path = output_dir / "quantize_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Quantization complete!")
    logger.info(f"Output saved to: {output_dir}")
    logger.info(f"Quantization report: {report_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    quantize()
