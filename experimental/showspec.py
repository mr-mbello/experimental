#!/usr/bin/env python

import typer
import numpy as np
import enum
from collections.abc import Sequence
import pathlib
from typing import NamedTuple
import pandas as pd
import scipy.signal as ssig
import matplotlib.pyplot as plt


class _SignalT(str, enum.Enum):
    roll = "roll"
    pitch = "pitch"
    ax = "ax"
    ay = "ay"
    az = "az"

class _Scale(str, enum.Enum):
    linear = "linear"
    log = "log"
    rootpsd = "rootpsd"


def read(fp: pathlib.Path) -> pd.DataFrame:
    columns: list[str] = [e.value for e in _SignalT]
    return pd.read_csv(fp, header=None, names=columns) # type: ignore


class Spectum(NamedTuple):
    f: Sequence[float]
    t: Sequence[float]
    sxx: Sequence[Sequence[float]]


def create_spectrum(data: Sequence[float], fs: float) -> Spectum:
    return Spectum(*ssig.spectrogram(data, fs))


def show(*, spectrum: Spectum, name: str, scale: _Scale) -> None:
    match scale:
        case _Scale.linear: sxx = spectrum.sxx
        case _Scale.log: sxx = -20 * np.log10(spectrum.sxx)
        case _Scale.rootpsd: sxx = np.sqrt(spectrum.sxx)
    im = plt.pcolormesh(
        spectrum.t,
        spectrum.f,
        sxx
    )
    plt.colorbar(im)
    plt.title(name)
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.show()


def app(
    filename: pathlib.Path,
    signal: _SignalT,
    fs: float = 320,
    scale: _Scale = "linear"
) -> None:
    data = read(filename)
    show(
        name=signal.value,
        scale=scale,
        spectrum=create_spectrum(getattr(data, signal.value), fs),
    )


def main():
    typer.run(app)
