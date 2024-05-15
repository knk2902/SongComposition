import os
import requests
import zipfile
import subprocess

# URL to download FluidSynth zip file
fluidsynth_url = "https://github.com/FluidSynth/fluidsynth/releases/download/v2.3.5/fluidsynth-2.3.5-win10-x64.zip"

# Directory to extract FluidSynth zip file
fluidsynth_dir = "C:\\"

# Download FluidSynth zip file
response = requests.get(fluidsynth_url)
with open("fluidsynth.zip", "wb") as zip_file:
    zip_file.write(response.content)

# Extract FluidSynth zip file
with zipfile.ZipFile("fluidsynth.zip", "r") as zip_ref:
    zip_ref.extractall(fluidsynth_dir)

# Add FluidSynth directory to system PATH (optional)
os.environ["PATH"] += os.pathsep + fluidsynth_dir + "FluidSynth\\bin\\"

print(os.environ["PATH"])

# Verify installation
try:
    subprocess.run(["fluidsynth", "--version"], check=True)
    print("FluidSynth installation successful.")
except subprocess.CalledProcessError:
    print("FluidSynth installation failed.")

# Clean up downloaded files
os.remove("fluidsynth.zip")
