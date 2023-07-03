import glob
import json
import os
import sys

from torch import version as torch_version

from llamahat import cuda as cuda_ext

from .model import ExLlamaDeviceMap


class ExLlamaConfig:
    # Load config from Llama config.json

    def __init__(self, model_config_path):
        with open(model_config_path) as f:
            read_config = json.load(f)

        # Loaded/automatic settings
        # Note that the HF LlamaTokenizer doesn't seem to recognize these automatically
        self.bos_token_id = read_config["bos_token_id"]
        self.eos_token_id = read_config["eos_token_id"]
        self.pad_token_id = read_config["pad_token_id"]

        self.hidden_size = read_config["hidden_size"]
        self.initializer_range = read_config["initializer_range"]
        self.intermediate_size = read_config["intermediate_size"]
        self.num_attention_heads = read_config["num_attention_heads"]
        self.num_hidden_layers = read_config["num_hidden_layers"]
        self.num_attention_heads = read_config["num_attention_heads"]
        self.rms_norm_eps = read_config["rms_norm_eps"]
        self.vocab_size = read_config["vocab_size"]

        # Constant used for pretrained models, leave as is unless retraining
        self.rotary_embedding_base = 10000
        self.head_dim = self.hidden_size // self.num_attention_heads

        self.groupsize = None  # Autodetected
        self.act_order = False  # Autodetected
        self.empty_g_idx = False  # Autodetected

        # Required settings
        self.model_path = None
        self.device_map = ExLlamaDeviceMap(self.num_hidden_layers)

        # Optional settings
        self.max_seq_len = 2048  # Reduce to save memory. Can also be increased, ideally while also using compress_pos_emn and a compatible model/LoRA
        self.max_input_len = 2048  # Maximum length of input IDs in a single forward pass. Sequences longer than this will be processed in multiple steps
        self.max_attention_size = (
            2048**2
        )  # Sequences will be processed in chunks to keep the size of the attention weights matrix <= this
        self.compress_pos_emb = 1.0  # Increase to compress positional embeddings applied to sequence
        self.alpha_value = 1.0  # Alpha value for NTK RoPE scaling. Similar to compress_pos_emb, higher values increaste ctx but add Perplexity.
        self.gpu_peer_fix = False  # Apparently Torch can have problems transferring tensors directly one GPU to another sometimes. Enable this to expliticly move tensors via system RAM instead, where needed
        self.auto_map = (
            None  # List of floats with memory allocation in GB, per CUDA device, overrides device_map
        )
        # Tuning
        self.matmul_recons_thd = 8
        self.fused_mlp_thd = 2
        self.sdp_thd = 8
        self.fused_attn = True
        self.matmul_fused_remap = False
        self.rmsnorm_no_half2 = False
        self.rope_no_half2 = False
        self.matmul_no_half2 = False
        self.silu_no_half2 = False
        self.concurrent_streams = False

    # Copy tuning params to C++ extension
    def set_tuning_params(self):
        cuda_ext.set_tuning_params(
            self.matmul_recons_thd,
            self.fused_mlp_thd,
            self.sdp_thd,
            self.matmul_fused_remap,
            self.rmsnorm_no_half2,
            self.rope_no_half2,
            self.matmul_no_half2,
            self.silu_no_half2,
            self.concurrent_streams,
        )

    # Parse and set list of GPU VRAM allocations
    def set_auto_map(self, map_string):
        if map_string is None:
            self.auto_map = None
        else:
            self.auto_map = [float(alloc) for alloc in map_string.split(",")]

    def calculate_rotary_embedding_base(self):
        self.rotary_embedding_base = self.rotary_embedding_base * self.alpha_value ** (
            self.head_dim / (self.head_dim - 2)
        )


def add_args(parser):
    parser.add_argument("-t", "--tokenizer", type=str, help="Tokenizer model path")
    parser.add_argument("-c", "--config", type=str, help="Model config path (config.json)")
    parser.add_argument("-m", "--model", type=str, help="Model weights path (.pt or .safetensors file)")
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        help="Path to directory containing config.json, model.tokenizer and * .safetensors",
    )

    parser.add_argument(
        "-gs",
        "--gpu_split",
        type=str,
        help="Comma-separated list of VRAM (in GB) to use per GPU device for model layers, e.g. -gs 20,7,7",
    )
    parser.add_argument("-l", "--length", type=int, help="Maximum sequence length", default=2048)
    parser.add_argument(
        "-cpe",
        "--compress_pos_emb",
        type=float,
        help="Compression factor for positional embeddings",
        default=1.0,
    )

    parser.add_argument(
        "-gs",
        "--gpu_split",
        type=str,
        help="Comma-separated list of VRAM (in GB) to use per GPU device for model layers, e.g. -gs 20,7,7",
    )
    parser.add_argument("-l", "--length", type=int, help="Maximum sequence length", default=2048)
    parser.add_argument(
        "-cpe",
        "--compress_pos_emb",
        type=float,
        help="Compression factor for positional embeddings",
        default=1.0,
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        help="alpha for context size extension via embedding extension",
        default=1.0,
    )

    parser.add_argument(
        "-gpfix", "--gpu_peer_fix", action="store_true", help="Prevent direct copies of data between GPUs"
    )

    parser.add_argument(
        "-mmrt",
        "--matmul_recons_thd",
        type=int,
        help="No. rows at which to use reconstruction and cuBLAS for quant matmul. 0 = never, 1 = always",
        default=8,
    )
    parser.add_argument(
        "-fmt",
        "--fused_mlp_thd",
        type=int,
        help="Maximum no. of rows for which to use fused MLP. 0 = never",
        default=2,
    )
    parser.add_argument(
        "-sdpt",
        "--sdp_thd",
        type=int,
        help="No. rows at which to switch to scaled_dot_product_attention. 0 = never, 1 = always",
        default=8,
    )
    parser.add_argument(
        "-mmfr", "--matmul_fused_remap", action="store_true", help="Fuse column remapping in Q4 matmul kernel"
    )
    parser.add_argument("-nfa", "--no_fused_attn", action="store_true", help="Disable fused attention")

    parser.add_argument(
        "-rnnh2", "--rmsnorm_no_half2", action="store_true", help="Don't use half2 in RMS norm kernel"
    )
    parser.add_argument(
        "-rpnh2", "--rope_no_half2", action="store_true", help="Don't use half2 in RoPE kernel"
    )
    parser.add_argument(
        "-mmnh2", "--matmul_no_half2", action="store_true", help="Don't use half2 in Q4 matmul kernel"
    )
    parser.add_argument(
        "-snh2", "--silu_no_half2", action="store_true", help="Don't use half2 in SiLU kernel"
    )
    parser.add_argument(
        "-nh2", "--no_half2", action="store_true", help="(All of the above) disable half2 in all kernela"
    )
    parser.add_argument(
        "-fh2", "--force_half2", action="store_true", help="Force enable half2 even if unsupported"
    )
    parser.add_argument(
        "-cs", "--concurrent_streams", action="store_true", help="Use concurrent CUDA streams"
    )


def post_parse(args):
    if args.no_half2 or torch_version.hip and not args.force_half2:
        args.rmsnorm_no_half2 = True
        args.rope_no_half2 = True
        args.matmul_no_half2 = True
        args.silu_no_half2 = True


def get_model_files(args):
    """Get model files from --directory"""
    if args.directory is not None:
        args.tokenizer = os.path.join(args.directory, "tokenizer.model")
        args.config = os.path.join(args.directory, "config.json")
        st_pattern = os.path.join(args.directory, "*.safetensors")
        st = glob.glob(st_pattern)
        if len(st) == 0:
            print(f" !! No files matching {st_pattern}")
            sys.exit()
        if len(st) > 1:
            print(f" !! Multiple files matching {st_pattern}")
            sys.exit()
        args.model = st[0]
    else:
        if args.tokenizer is None or args.config is None or args.model is None:
            print(" !! Please specify either -d or all of -t, -c and -m")
            sys.exit()


def print_options(args, extra_options=None):
    print_opts = []
    if args.gpu_split is not None:
        print_opts.append(f"gpu_split: {args.gpu_split}")
    if args.gpu_peer_fix:
        print_opts.append("gpu_peer_fix")

    if extra_options is not None:
        print_opts += extra_options

    print(f" -- Tokenizer: {args.tokenizer}")
    print(f" -- Model config: {args.config}")
    print(f" -- Model: {args.model}")
    print(f" -- Sequence length: {args.length}")
    if args.compress_pos_emb != 1.0:
        print(f" -- RoPE compression factor: {args.compress_pos_emb}")

    if args.alpha != 1.0:
        print(f" -- RoPE alpha factor: {args.alpha}")

    print(" -- Tuning:")
    print(
        f" -- --matmul_recons_thd: {args.matmul_recons_thd}"
        + (" (disabled)" if args.matmul_recons_thd == 0 else "")
    )
    print(f" -- --fused_mlp_thd: {args.fused_mlp_thd}" + (" (disabled)" if args.fused_mlp_thd == 0 else ""))
    print(f" -- --sdp_thd: {args.sdp_thd}" + (" (disabled)" if args.sdp_thd == 0 else ""))
    if args.matmul_fused_remap:
        print(" -- --matmul_fused_remap")
    if args.no_fused_attn:
        print(" -- --no_fused_attn")
    if args.rmsnorm_no_half2:
        print(" -- --rmsnorm_no_half2")
    if args.rope_no_half2:
        print(" -- --rope_no_half2")
    if args.matmul_no_half2:
        print(" -- --matmul_no_half2")
    if args.silu_no_half2:
        print(" -- --silu_no_half2")
    if args.concurrent_streams:
        print(" -- --concurrent_streams")

    print(f" -- Options: {print_opts}")


def make_config(args) -> ExLlamaConfig:
    """Build ExLlamaConfig from args"""
    config = ExLlamaConfig(args.config)
    config.model_path = args.model

    config.max_seq_len = args.length
    config.compress_pos_emb = args.compress_pos_emb
    config.set_auto_map(args.gpu_split)
    config.gpu_peer_fix = args.gpu_peer_fix
    config.alpha_value = args.alpha
    config.calculate_rotary_embedding_base()

    config.matmul_recons_thd = args.matmul_recons_thd
    config.fused_mlp_thd = args.fused_mlp_thd
    config.sdp_thd = args.sdp_thd
    config.matmul_fused_remap = args.matmul_fused_remap
    config.fused_attn = not args.no_fused_attn

    config.rmsnorm_no_half2 = args.rmsnorm_no_half2
    config.rope_no_half2 = args.rope_no_half2
    config.matmul_no_half2 = args.matmul_no_half2
    config.silu_no_half2 = args.silu_no_half2
    config.concurrent_streams = args.concurrent_streams

    return config


# Print stats after loading model
def print_stats(model):
    print(f" -- Groupsize (inferred): {model.config.groupsize or 'None'}")
    print(f" -- Act-order (inferred): {'yes' if model.config.act_order else 'no'}")
    if model.config.empty_g_idx:
        print(" !! Model has empty group index (discarded)")
