# Author of this document
Milad LEYLI ABADI (IRT SystemX)

# Server Organisation
This is a suggestion for organisation of our share space on Exaion GPU server

Each contributor (SystemX, RTE, Nvidia) has its own folder under the root (~/):
- SYSTEMX
- RTE
- NVIDIA

The codabench configs and the corresponding version of LIPS framework repository could coexist in root as well,
to simplify the configurations. 

Each contributor could clone independently the LIPS package and install required tools into its own folder and 
to experiment without disturbing the shared space.  

# Discussion channel
Don't hesitate to join the Exaion channel for the discussions concerning the GPU server (https://chat.exaion.com)
Don't hesitate to join the Discord channel for the discussions concerning the LIPS platform and to get the notes
concerning the last updates (https://discord.gg/TYSrRg4m)

# Intruction to request a new acess
1. Add the Public SSH key of the new user in ~/.ssh/authorized_keys 
2. Send the public ip address via the channel to exaion.

# LIPS package
The last version of LIPS package could be cloned from [here](https://github.com/Mleyliabadi/LIPS).
The main branch contains the last stable version.

# Tensorflow and Torch installation
For pytorch use the following command, that works with Cuda 11.7:
```commandline
pip install torch --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu116
```

For tensorflow, install the last version :
```commandline
pip install tensorflow==2.9.0
```

# GPU resource usage
In order to use only one GPU among four and not to occupy all the resources, you can do the following:

## Tensorflow
At the begining of your code add the following lines
```python
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        # Visible devices must be set at program startup
        print(e)
```

Verify if the changes took effect:
```python
tf.config.experimental.get_visible_devices()
```
which should give:
```bash
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'),
 PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## Pytorch
You set the index of the GPU using the `device` argument of `TorchSimulator` class. 
```python
from lips.augmented_simulators.torch_models.fully_connected import TorchFullyConnected
from lips.augmented_simulators.torch_simulator import TorchSimulator
from lips.dataset.scaler import StandardScaler

torch_sim = TorchSimulator(name="torch_fc",
                           model=TorchFullyConnected,
                           scaler=StandardScaler,
                           log_path=LOG_PATH,
                           device="cuda:0", # use the first GPU 
                           seed=42,
                           bench_config_path=BENCH_CONFIG_PATH,
                           bench_config_name="Benchmark1",
                           sim_config_path=SIM_CONFIG_PATH / "torch_fc.ini",
                           sim_config_name="DEFAULT" 
                          )
```

# Install GraphViz
The following commands let install the graphviz package and to use it with keras:

```commandline
source /path_to_venv/bin/activate
pip install pydot-ng
sudo apt-get install graphviz
```

Once installed properly, you could visualize and save the neural network architecture, when using Tensorflow:
```python
from lips.augmented_simulators.tensorflow_models import TfFullyConnected
from lips.dataset.scaler import StandardScaler

tf_fc = TfFullyConnected(name="tf_fc",
                         bench_config_path=BENCH_CONFIG_PATH,
                         bench_config_name="Benchmark1",
                         sim_config_path=SIM_CONFIG_PATH / "tf_fc.ini",
                         sim_config_name="DEFAULT",
                         scaler=StandardScaler,
                         log_path=LOG_PATH)

# Train it first
tf_fc.train(train_dataset=benchmark1.train_dataset,
            val_dataset=benchmark1.val_dataset,
            epochs=10
           )

# Export the architecture to .png file
tf_fc.plot_model(path=".", file_name="tf_fc")
```

# Use Jupyter Notebooks
This section explaines how to use jupyter notebooks from remote on your local machine.

## This steps are on the GPU server
The first step to be able to use the virtual environment in notebook is to create a virtual environment,
and install it as kernel in jupyter notebook kernelspec:

If the virtual environment is already installed and is set in jupyter kernelspec:
```commandline
source venv/lips/bin/activate
```

If you want to use your own virtual environment from scratch:
```commandline
python -m pip install virtualenv
python -m virtualenv lips
source lips/bin/activate
python -m pip install ipython
python ipykernel install --user --name lips
```

To be able to use jupyter notebook, the first step is to run a jupyter using a specific port:
```commandline
jupyter notebook --no-browser --port=8080
```
!!! Attention: select an available port. the ports could already taken by the other users or the system itself

To see the taken ports, you can use:
```commandline
sudo lsof -i -P -n | grep LISTEN
```

When launching notebooks you can also add the `&` symbol at the end of your command, to run the notebook in 
background and to keep using the command line interface.

```commandline
jupyter notebook --no-browser --port=8080 &
```

## These steps are on local machine
Afterwards, on the local machine, you should map your localhost to the localhost of the server as:
```commandline
ssh -N -f -L localhost:8080:localhost:8080 ubuntu@91.239.56.172 -p 45122
```

Then, you can use the Notebook like a charm using your navigator and the following URL:
http://localhost:8080/

# Server information
I have updated the cuda driver to the last version 11.7. 
To see the cuda version, use the command:

```commandline
nvcc --version
```

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Tue_May__3_18:49:52_PDT_2022
Cuda compilation tools, release 11.7, V11.7.64
Build cuda_11.7.r11.7/compiler.31294372_0
``` 

# Contact information
[Milad LEYLI ABADI](milad.leyli-abadi@irt-systemx.fr) IRT SystemX
