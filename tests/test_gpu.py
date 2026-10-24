"""GPU detection: AMD integrated Radeon graphics must count, and Windows never asks MediaPipe for the GPU."""
import sys
import types

from memecam import tracking


def fake_winreg(adapters):
    """A winreg stand-in whose display-adapter class key holds `adapters` [(DriverDesc, MatchingDeviceId)]."""
    w = types.ModuleType("winreg")
    w.HKEY_LOCAL_MACHINE = 0

    class Key:
        def __init__(self, values=None):
            self.values = values

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def open_key(parent, sub):
        if isinstance(parent, Key):
            return Key(adapters[int(sub)])
        return Key()

    def enum_key(key, i):
        if i >= len(adapters):
            raise OSError
        return f"{i:04d}"

    def query(key, name):
        desc, dev = key.values
        return ({"DriverDesc": desc, "MatchingDeviceId": dev}[name], 1)

    w.OpenKey, w.EnumKey, w.QueryValueEx = open_key, enum_key, query
    return w


def test_windows_amd_igpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "winreg", fake_winreg([
        ("AMD Radeon(TM) Graphics", "PCI\\VEN_1002&DEV_1638&SUBSYS_1"),
        ("Microsoft Basic Display Adapter", "root\\basicdisplay"),
    ]))
    monkeypatch.setattr(tracking, "linux_gpus", lambda: [])
    assert tracking.windows_gpus() == [("amd", "AMD Radeon(TM) Graphics")]
    assert tracking.discrete_gpu() == "amd"


def test_windows_vendor_from_name(monkeypatch):
    monkeypatch.setitem(sys.modules, "winreg", fake_winreg([("AMD Radeon Vega 8 Graphics", "")]))
    assert tracking.windows_gpus() == [("amd", "AMD Radeon Vega 8 Graphics")]


def test_windows_stays_on_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "winreg", fake_winreg([("AMD Radeon(TM) 780M", "pci\\ven_1002&dev_15bf")]))
    monkeypatch.setattr(tracking.sys, "platform", "win32")
    delegate, why = tracking.pick_delegate("auto")
    assert delegate == "cpu" and "AMD Radeon(TM) 780M" in why
    assert tracking.pick_delegate("gpu")[0] == "cpu"


def test_linux_amd_apu(monkeypatch):
    monkeypatch.setattr(tracking, "linux_gpus", lambda: ["amd"])
    monkeypatch.setattr(tracking.sys, "platform", "linux")
    assert tracking.pick_delegate("auto")[0] == "gpu"
    monkeypatch.setattr(tracking, "linux_gpus", lambda: ["intel"])
    monkeypatch.setattr("shutil.which", lambda _: None)
    assert tracking.pick_delegate("auto")[0] == "cpu"
