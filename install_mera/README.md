# How to install MERA Software Stack

For quick installation, please run a single script below, it will run all the steps and automatically install everything.

``` bash
source install_all_steps.sh
```

For a more customized installation, please follow the steps below.

## Step 1: install OS system dependencies

Note: user only needs to do this once. requires sudo.

``` bash
./install_os_dependencies.sh
```

## Step 2: create new virtual environment

* below script will automatically creates a new empty environment. Users can opt to create their own without running this.
* once created, user can run `source start.sh` to activate the environment.

```bash
./create_virtual_env.sh
source start.sh
```

## Step 3: install mera and its dependencies

this will install the main mera software library

``` bash
./install_mera_and_python_dependencies.sh
```

## Step 4: [optional] install mera-models and its dependencies

for users who want a more convenient way to work with models on huggingface, or run some examples showcasing those methods, `mera-models` library should be installed.

``` bash
./install_mera_models_and_python_dependencies.sh
```

## Step 5: [optional] install mera-visualizer and its dependencies

for users who want visualization of model graph.

``` bash
./install_mera_visualizer_and_python_dependencies.sh
```
