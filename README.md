# MERA software package

For detailed steps on installation, please download the Mera Installation and User Manual from the following link on our developer zone. https://www.edgecortix.com/en/developers/mera-installation-and-user-manual

For information of current and past releases, please see [RELEASE_NOTES.md](RELEASE_NOTES.md)

## Step 1 - Install MERA

To install MERA move to the following directory and follow the instructions on the README.md file:

```bash
cd install_mera/
cat README.md
```



## Step 2 - Initialize SAKURA-II Board

At the very beginning, or after each reboot, we need to load the required SAKURA-II board Linux kernel driver as well as some daemon processes. 

Run the following command to initialize the board:

```bash
cd initialize_sakura_ii/

chmod +x ./setup.sh
./setup.sh
```

After this command, the board should be up and running, ready to be used to run demos and perform inference.

Note: This procedure is required **only once after each reboot** of the system. For more details, refer to `initialize_sakura_ii/README.md` file.



## Step 3 - Run examples

For examples showcasing various use cases, please go to:
```bash
cd examples/
```
All examples assume that MERA is installed in the current virtual environment.

And follow the steps in the README file

<hr>

From the next reboot onward, follow these steps:
  1. Initialize SAKURA-II Board
  2. Activate MERA installed virtual environment
  3. Run inference
