
# EfficientNet Lite Classification Demo

The Jupiter notebook file `EfficientNetDemo.ipyb` provided an illustrated example of how to use MERA. (please see Appendix)

For more details, please look at repository: [tensorflow/tpu](https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/lite/README.md).

## How to get weight files

```bash
cd source_model_files
./get_weights.sh
```

## Model compilation and deployment with MERA software stack
This creates a deployment folder for `ip` to run on the card
```bash
python deploy.py --target ip
```
If you wanted to deploy for `Simulator` then issue the following command instead.
```bash
python deploy.py --target simulator
```

## Inference

To run the model on the card with target ip

```bash
python demo_model.py --target ip
```

To run the model on simulator

```bash
python demo_model.py --target simulator
```

# Appendix

## How to use Jupyter Notebook

```bash
jupyter notebook --ip $IPV4 --port 8080 --no-browser
```

Then click on the provided link on the output of the previous command.
Finally from the Jupyter notebook web interface, click on the `EfficientNetDemo.ipynb` notebook.

