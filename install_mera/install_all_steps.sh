#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

sudo echo Installing...  # this forces immediate password entry

printf "\n\e[1;32m%s\e[0m\n\n" "Step 1 -- install_os_dependencies"
sleep 1s
./install_os_dependencies.sh

printf "\n\e[1;32m%s\e[0m\n\n" "Step 2 -- create_virtual_env"
sleep 1s
./create_virtual_env.sh
source ./start.sh
printf "%s\n" "[Loaded virtual env]"

printf "\n\e[1;32m%s\e[0m\n\n" "Step 3 -- install_mera_and_python_dependencies"
sleep 1s
./install_mera_and_python_dependencies.sh

printf "\n\e[1;32m%s\e[0m\n\n" "Step 4 -- install_mera_models_and_python_dependencies"
sleep 1s
./install_mera_models_and_python_dependencies.sh

printf "\n\e[1;32m%s\e[0m\n\n" "Step 5 -- install_mera_visualizer_python_dependencies"
sleep 1s
./install_mera_visualizer_and_python_dependencies.sh

printf "\n\e[1;32m%s\e[0m\n\n" "Installation complete"

set +e +u +o pipefail
