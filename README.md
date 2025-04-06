# SHVIT - SegFormer
Anweisungen zur Nutzung des Codes sowie Details zur Ausführung der relevanten Notebook-Datei

## Verzeichnisstruktur
Der relevante Code befindet sich im folgenden Pfad:

`PDF_Exchanger/SHVIT/Python/04 - SHVIT_Segformer`

Wichtige Datei:
- SegFormer_3.ipynb: Dieses Notebook wird verwendet, um das Modell zu trainieren und Ergebnisse zu generieren

======================================================================

To facilitate the execution of Linux-based software and tools in a Windows environment, Windows
Subsystem for Linux (WSL) was set up on a Windows 11 machine. The following steps outline
the process of setting up WSL, installing Miniconda, and creating a virtual environment to run
PyTorch 2.60 with GPU support.
After downloading and installing Ubuntu 24.04 using Microsoft Store the following steps have to
be carried out. Before step 4 is executed Miniconda had to be downloaded (https://www.anaconda.com/download/success):

```
sudo apt update -y
sudo apt upgrade -y
cd /mnt/c/Users/USERNAME/Downloads/
bash Miniconda3-latest-Linux-x86_64.sh
cd ~
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
exit
```

The installation of PyTorch is very easy and does not require additional installation of CUDA
drivers or other dependencies, unlike TensorFlow. PyTorch comes with built-in CUDA support (https://pytorch.org/get-started/locally/),
meaning it can be simply installed using pip, and it will automatically handle the necessary CUDA
libraries.

```
conda create -n torch260 python=3.12
conda activate torch260
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
conda install -c conda-forge opencv matplotlib tqdm seaborn pandas plotly lightning
pip install torchsummary torchviz pydicom slicerio
sudo apt-get install graphviz
cd Documents/Python/shoes_segformer/Software/
pip install PythonTools-3.7.0-py2.py3-none-any.whl
conda install conda-forge::libsqlite --force-reinstall
conda install conda-forge::sqlite --force-reinstall
```

After setting up PyTorch, essential libraries like `matplotlib`, `tqdm`, `seaborn`, etc. needed for
running the SegFormer scripts have to be installed. Additionally, `pydicom` and `slicerio` were
included for handling the annotation les for the shoe les. To extract data from the original shoe
volume data from the files in `.rek` dataformat, the `PythonTools` software was required.
While running the scripts in VSCode, the kernel occasionally crashed. This issue was resolved by
reinstalling the `sqlite` and `libsqlite` packages at the end (https://stackoverflow.com/a/79484466/27900239).




