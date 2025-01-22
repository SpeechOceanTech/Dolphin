# encoding: utf8

import os
import logging
import argparse
from argparse import Namespace
from pathlib import Path
from distutils.util import strtobool
from typing import Union, Optional, Tuple

import torch

from .audio import load_audio
from .model import DolphinSpeech2Text

logger = logging.getLogger("dolphin")


def str2bool(value: str) -> bool:
    return bool(strtobool(value))


def parser_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=str, help="audio file path")
    parser.add_argument("--model", type=str, default="small", help="model name")
    parser.add_argument("--model_dir", type=Path, required=True, help="model checkpoint download diretory")
    parser.add_argument("--lang_sym", type=str, default=None, help="language symbol (e.g. <zh>)")
    parser.add_argument("--region_sym", type=str, default=None, help="regiion symbol (e.g. <CN>)")
    parser.add_argument("--dtype", type=str, default="float32", help="data type (default: float32)")
    parser.add_argument("--device", type=str, default=None, help="torch device (default: None)")
    parser.add_argument("--normalize_length", type=str2bool, default=False, help="whether to normalize length (default: false)")
    parser.add_argument("--padding_speech", type=str2bool, default=True, help="whether padding speech to 30 seconds (default: true)")
    parser.add_argument("--predict_time", type=str2bool, default=True, help="whether predict timestamp (default: true)")
    parser.add_argument("--beam_size", type=int, default=5, help="number of beams in beam search (default: 5)")
    parser.add_argument("--maxlenratio", type=float, default=0.0, help="Input length ratio to obtain max output length (default: 0.0)")

    args = parser.parse_args()
    return args


def load_model(
    model_name: str,
    model_dir: Path,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs
) -> DolphinSpeech2Text:
    """
    Load DolphinSpeech2Text model.

    Args:
        model_name: model name (e.g. small)
        model_dir: model download directory
        device: the pytorch device

    Returns:
        DolphinSpeech2Text instance
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not isinstance(model_dir, Path):
        model_dir = Path(model_dir)

    model_file = model_dir / f"{model_name}.pt"
    train_cfg = model_dir / "config.yaml"
    if not model_file.exists():
        logger.error(f"model {model_name} not found.")
        raise Exception(f"model {model_name} not found, please download the model first.")

    model = DolphinSpeech2Text(
        s2t_train_config=train_cfg,
        s2t_model_file=model_file,
        device=device,
        task_sym=kwargs.get("task_sym", "<asr>"),
        predict_time=kwargs.get("predict_time", True),
        **kwargs,
    )
    return model


def transcribe(args: Namespace) -> Tuple[str, str]:
    """
    Transcribe audio to text.

    Args:
        args: the command line parameters

    Returns:
        result (text, text_nospecial)
    """
    model_name = args.model
    model = load_model(model_name, args.model_dir, args.device)
    waveform = load_audio(args.audio)

    result = model(
        speech=waveform,
        lang_sym=args.lang_sym,
        region_sym=args.region_sym,
        predict_time=args.predict_time,
        padding_speech=args.padding_speech
    )

    logger.info(f"decode result, text: {result['text']}")
    return result


def cli():
    logging.basicConfig(level=logging.INFO, format=logging.BASIC_FORMAT)

    # filter framework interanl logs
    logging.getLogger("espnet").setLevel(logging.ERROR)
    logging.getLogger("root").setLevel(logging.ERROR)
    logging.getLogger("dolphin").setLevel(logging.INFO)

    args = parser_args()
    transcribe(args)


if __name__ == "__main__":

    cli()
