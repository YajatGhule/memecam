# Meme Cam installer for Windows (PowerShell). Linux / macOS: install.sh.
#
#   powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/YajatGhule/memecam/main/install.ps1 | iex"
#
# Installs uv if needed (it brings its own Python 3.12), clones the repo (git, or a zip download without
# git), installs the dependencies, downloads the MediaPipe models and adds Start-menu shortcuts plus
# memecam.cmd / memecamv2.cmd launchers. Re-run it to update.
# Overrides: $env:MEMECAM_DIR (install dir), $env:MEMECAM_REF (branch / tag).
$ErrorActionPreference = "Stop"

$Repo = "https://github.com/YajatGhule/memecam"
$Ref = if ($env:MEMECAM_REF) { $env:MEMECAM_REF } else { "main" }
$Dir = if ($env:MEMECAM_DIR) { $env:MEMECAM_DIR } else { Join-Path $env:LOCALAPPDATA "memecam" }

function Say($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }

# ---- 1. uv ----
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) {
    $candidate = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
    if (Test-Path $candidate) { $uv = $candidate }
}
if (-not $uv) {
    Say "installing uv (https://docs.astral.sh/uv)"
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $uv = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
    if (-not (Test-Path $uv)) { throw "uv install failed" }
}

# ---- 2. source ----
$git = Get-Command git -ErrorAction SilentlyContinue
if (Test-Path (Join-Path $Dir ".git")) {
    Say "updating $Dir"
    git -C $Dir fetch --depth 1 origin $Ref
    git -C $Dir checkout -q -B $Ref FETCH_HEAD
} elseif ($git) {
    Say "cloning into $Dir"
    git clone --depth 1 --branch $Ref "$Repo.git" $Dir
} else {
    Say "downloading $Repo ($Ref) as a zip (no git found)"
    $zip = Join-Path $env:TEMP "memecam.zip"
    $tmp = Join-Path $env:TEMP "memecam-src"
    Invoke-WebRequest "$Repo/archive/refs/heads/$Ref.zip" -OutFile $zip
    if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
    Expand-Archive $zip $tmp
    $src = Get-ChildItem $tmp | Select-Object -First 1
    New-Item -ItemType Directory -Force $Dir | Out-Null
    Copy-Item (Join-Path $src.FullName "*") $Dir -Recurse -Force
    Remove-Item $zip, $tmp -Recurse -Force
}

# ---- 3. dependencies + models ----
Say "installing Python 3.12 + dependencies (MediaPipe, OpenCV, ...)"
& $uv sync --project $Dir --no-dev --python 3.12
if ($LASTEXITCODE) { throw "uv sync failed" }
Say "downloading the MediaPipe models"
& $uv run --project $Dir --no-dev python -c "from memecam.tracking import MODELS, model_path; [model_path(m) for m in MODELS]"
if ($LASTEXITCODE) { throw "model download failed" }

# ---- 4. launchers ----
$bin = Join-Path $env:USERPROFILE ".local\bin"
New-Item -ItemType Directory -Force $bin | Out-Null
$menu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
$shell = New-Object -ComObject WScript.Shell
foreach ($exe in "memecam", "memecamv2") {
    Set-Content (Join-Path $bin "$exe.cmd") "@`"$uv`" run --project `"$Dir`" --no-dev --quiet $exe %*"
    $name = if ($exe -eq "memecam") { "Meme Cam" } else { "Meme Cam v2" }
    $lnk = $shell.CreateShortcut((Join-Path $menu "$name.lnk"))
    $lnk.TargetPath = $uv
    $lnk.Arguments = "run --project `"$Dir`" --no-dev --quiet $exe"
    $lnk.WorkingDirectory = $Dir
    $lnk.Save()
}

Say "done"
Write-Host "  run:     Meme Cam in the Start menu, or  memecam  / memecamv2  in a new terminal"
Write-Host "  keys:    d markers, f memes, i debug, r record, c recalibrate, q quit"
Write-Host "  update:  re-run this installer"
if (-not (($env:PATH -split ";") -contains $bin)) {
    Write-Host "  note:    $bin is not on PATH yet; open a new terminal after installing uv, or use the Start menu" -ForegroundColor Yellow
}
Write-Host "  camera:  allow desktop apps in Settings > Privacy & security > Camera"
