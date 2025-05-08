# Resnet50 Demo

This is a resnet50 example. Below is a basic compilation and inference.
More indepth-explanation is inside each python file.

## Model compilation and inference with MERA software stack

### Compilation:

The code to export the model in INT8 torchscript and compile the model is as follows.
```bash
python deploy.py --target ip
```

### High-Performance Compilation Mode

By default, the Scheduling mode is set to `Simple`, prioritizing quick compilation times. To optimize latency, you can switch the Scheduling mode to `Performance`.

Below is an example of command usage:

For demonstration purposes, the number of iterations is set to `1000`. However, for optimal results, we recommend setting it to `32000`.

```bash
python deploy.py --target ip --out_dir "deploy_resnet50_perf_sched" \
--build_cfg "{'scheduler_config': {'mode': 'Performance', 'main_scheduling_iterations': 1000, 'progress_bars': true}}"
```

Please see the [measure_latency folder](../measure_latency) for latency measurements after finishing the deployment.

### Inference:

To run the model on the card with target `ip`

```bash
python demo_model.py --target ip
```
