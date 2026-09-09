"""Post-training: merge the LoRA into the base model, save, optionally export GGUF.

Merging a QLoRA adapter needs the base model in a non-quantised dtype, which is
why merging loads the base weights again in bf16/fp16 instead of reusing the
4-bit training model.
"""

from __future__ import annotations

from pathlib import Path

from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from .config import TrainConfig
from .model import resolve_dtype


def merge_adapter(
    base_model_id: str,
    adapter_dir: str | Path,
    out_dir: str | Path,
    *,
    track: str = "text",
    dtype: str = "auto",
    trust_remote_code: bool = True,
):
    """Load base + adapter, merge, and write a standalone model directory."""
    from peft import PeftModel

    torch_dtype = resolve_dtype(dtype)
    loader = AutoModelForImageTextToText if track == "vl" else AutoModelForCausalLM
    print(f"[merge] loading base {base_model_id} in {torch_dtype} ...")
    model = loader.from_pretrained(
        base_model_id,
        dtype=torch_dtype,
        device_map="cpu",  # merging on CPU avoids VRAM spikes on the 15 GB T4
        trust_remote_code=trust_remote_code,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=trust_remote_code)
    tokenizer.save_pretrained(out_dir)
    print(f"[merge] merged model written to {out_dir}")
    return out_dir


def push_directory(path: str | Path, repo_id: str, *, private: bool = True) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(path), repo_id=repo_id)
    print(f"[push] uploaded {path} -> {repo_id}")


def export_gguf(
    model_dir: str | Path,
    outfile: str | Path | None = None,
    *,
    quant: str = "q4_k_m",
    llama_cpp_dir: str | Path = "llama.cpp",
) -> Path:
    """Convert a merged HF model to GGUF using llama.cpp's converter.

    Requires network access (git clone of llama.cpp) plus its Python deps.
    Kept deliberately thin: llama.cpp moves fast and pinning it here would rot.
    """
    import subprocess

    llama_cpp_dir = Path(llama_cpp_dir)
    if not llama_cpp_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", str(llama_cpp_dir)],
            check=True,
        )
        subprocess.run(
            ["pip", "install", "-q", "-r", str(llama_cpp_dir / "requirements.txt")], check=True
        )

    model_dir = Path(model_dir)
    outfile = Path(outfile) if outfile else model_dir.with_suffix("") / f"{model_dir.name}-{quant}.gguf"
    subprocess.run(
        [
            "python",
            str(llama_cpp_dir / "convert_hf_to_gguf.py"),
            str(model_dir),
            "--outfile",
            str(outfile),
            "--outtype",
            quant,
        ],
        check=True,
    )
    print(f"[gguf] wrote {outfile}")
    return outfile


def run_export(cfg: TrainConfig, adapter_dir: str | Path) -> None:
    export = cfg.export
    merged_dir = None
    if export.merge_adapter or export.save_gguf:
        merged_dir = export.merged_dir or str(Path(cfg.training.output_dir) / "merged")
        merge_adapter(
            cfg.model.id,
            adapter_dir,
            merged_dir,
            track=cfg.model.track,
            trust_remote_code=cfg.model.trust_remote_code,
        )
    if export.push_to_hub and export.hub_repo_id:
        push_directory(merged_dir or adapter_dir, export.hub_repo_id)
    if export.save_gguf:
        if merged_dir is None:
            raise ValueError("save_gguf requires merge_adapter=true")
        export_gguf(merged_dir, export.gguf_outfile, quant=export.gguf_quant)
