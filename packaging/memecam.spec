# PyInstaller spec: one folder with memecam(.exe) + its runtime (_internal/) + assets, config and models.
#   uv run --with pyinstaller pyinstaller packaging/memecam.spec --noconfirm
# then copy dist/memecam/memecam(.exe) to memecamv2(.exe) in the same folder (see .github/workflows).
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

root = Path(SPECPATH).parent
datas = [(str(root / "assets"), "assets"),
         (str(root / "memecam" / "config.toml"), "memecam"),
         (str(root / "memecam" / "config_v2.toml"), "memecam")]
datas += [(str(p), "models") for p in (root / "models").glob("*.task")]
datas += collect_data_files("mediapipe")
binaries = collect_dynamic_libs("mediapipe")

a = Analysis([str(root / "packaging" / "launch.py")], pathex=[str(root)], binaries=binaries, datas=datas,
             hiddenimports=["memecam.audio", "sounddevice"], excludes=["tkinter", "pytest"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="memecam", console=True,
          icon=None, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="memecam", upx=False)
