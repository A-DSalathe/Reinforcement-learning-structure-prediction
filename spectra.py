#!/usr/bin/env python3
from __future__ import annotations
import ase
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from ase import Atoms
from ase.vibrations import Infrared
from collections.abc import Callable, Sequence

# We will use tblite as a semiempirical code to model spectra
from tblite.ase import TBLite as XTB3


def plot_spectrum(
    x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, ax: Axes, **plot_kwargs
) -> Axes:  # pragma: no cover
    """Plot spectrum."""
    ax.plot(x, y, **plot_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(0, np.max(y) * 1.05)
    return ax


def spectra_from_arrays(
    positions: np.ndarray = np.array([[0, 0, 0]]),
    chemical_symbols: list = ["B"],
    name: str = "test",
):
    assert len(chemical_symbols) == positions.shape[0]
    nanoparticle = Atoms(chemical_symbols, positions)
    nanoparticle.calc = XTB3(method="GFN2-xTB", max_iterations=1000)
    ir = Infrared(nanoparticle)
    ir.run()

    energy_range, spectrum = ir.get_spectrum(
        start=0, end=1000, width=10, normalize=True
    )

    ir.write_spectra(f"{name}.dat", start=0, end=1000, width=10)
    ax = plt.axes(label="IR")
    ax = plot_spectrum(
        x=energy_range,
        y=spectrum,
        xlabel=r"$\tilde\nu$ / (cm$^{-1}$)",
        ylabel=r"IR intensity (a.u.)",
        ax=ax,
    )
    plt.savefig(f"{name}.png")

    # Might need to adapt this depending on os, but it helps
    shutil.rmtree("ir", ignore_errors=True)


def test():
    spectra_from_arrays()


if __name__ == "__main__":
    nanoparticle = ase.io.read(sys.argv[1])
    print(f"Running on {sys.argv[1]}...")
    nanoparticle.calc = XTB3(method="GFN2-xTB", max_iterations=1000)
    ir = Infrared(nanoparticle)
    ir.run()

    # The spectra is derived from frequencies and intensities (height and position of peaks)
    # It might be preferable to compute loss on these than on the processed spectra
    # since the refinement (so-called fold) is deterministic anyway
    # print(ir.summary())
    # print(f'Frequencies: {ir.get_frequencies()}')
    # print(f'IR_Intensities: {ir.get_energies()}')

    # From the frequencies and intensities (height and position of peaks),
    # get_spectrum uses a Lorentz/Guasian smear (or fold) to plot a spectra
    # The actual function is in
    # https://gitlab.com/ase/ase/-/blob/master/ase/vibrations/infrared.py?ref_type=heads
    # which calls the fold method from the base class Vibrations
    energy_range, spectrum = ir.get_spectrum(
        start=0, end=1000, width=10, normalize=True
    )

    # Note that the normalization (normalize=True) above
    # might imply a loss of information regarding peak height!
    # print(max(spectrum), min(spectrum))

    ir.write_spectra(f'{sys.argv[1].split(".")[0]}.dat', start=0, end=1000, width=10)
    ax = plt.axes(label="IR")
    ax = plot_spectrum(
        x=energy_range,
        y=spectrum,
        xlabel=r"$\tilde\nu$ / (cm$^{-1}$)",
        ylabel=r"IR intensity (a.u.)",
        ax=ax,
    )
    plt.savefig(f'{sys.argv[1].split(".")[0]}.png')
