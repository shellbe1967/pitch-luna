# -*- mode: python ; coding: utf-8 -*-
# PyInstaller の設定 (python -m PyInstaller build/PitchLuna.spec)
import os
from PyInstaller.utils.hooks import collect_all

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
OWN = [os.path.splitext(f)[0] for f in os.listdir(ROOT) if f.endswith(".py") and f not in ("launcher.py", "pitchluna_deps.py")]

datas, binaries, hidden = [], [], ["pitchluna_deps"]
for pkg in ("demucs", "torchcrepe", "librosa", "julius", "openunmix", "lazy_loader", "soundfile", "sounddevice",
            "audioread", "pygame"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hidden += h
    except Exception as e:
        print("collect_all skip", pkg, e)

a = Analysis(
    [os.path.join(ROOT, "launcher.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    excludes=OWN + ["matplotlib", "IPython", "jupyter", "notebook", "pytest", "tensorboard", "PyQt5", "PyQt6",
                    "PySide2", "PySide6", "sphinx", "black", "pandas.tests", "scipy.tests", "numpy.tests"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="PitchLuna",
    console=False,
    icon=os.path.join(ROOT, "assets", "icon.ico"),
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="PitchLuna", upx=False)
