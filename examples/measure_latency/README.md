# MERA Latency Measurement Tool

This tool measures and analyzes execution latency for MERA-deployed models on the SAKURA-2 platform.

## Usage

Basic usage:
```bash
python measure.py --model_path <path_to_deployed_model>
```

Recursive usage (measuring all `deploy_*` deployments):
```bash
python measure.py --model_path ../../examples/ --recursive
```

Options:
- `--model_path`: Path to deployed model directory (default: "./deployments")
- `--recursive`: Search recursively for models in subdirectories
- `--loop_cnt`: Number of iterations for averaging (default: 100)
- `--name`: Prefix for output file names
- `--run_forever`: Run a single deployment forever, for testing purpose
- `--row_fetch`: Enable row-fetching for latency measurement, only for LLM models (`huggingface_text_gen`)

## Output Metrics

The tool measures these latency components (in milliseconds):
- SetInput: Time taken to set input data
- PCIe DMA: PCIe Data transfer time
- DNA IP: IP execution time
- CPU_Reorder: Time for data reordering
- CPU_Ops: Additional CPU operations time
- EndToEnd: Total execution time (excluding SetInput)

## Example Output

Here is example of running measurement on the [resnet50 example folder](../resnet50) with this command

```
python measure.py --model_path ../../examples/resnet50/ --recursive
```

Results are saved as CSV and displayed in a formatted table:

### Note

* This was run on a PCIe card with X8 data transfer speed. Different configurations could have varying latency numbers. 
* PCIe DMA Data transfer is in FP32 format.
```
+----+------------------------------------+------------+------------+----------+-----------+-----------+-------------+
|    | Network                            |   SetInput |   PCIe DMA |   DNA IP |       CPU |   CPU Ops |    EndToEnd |
|    |                                    |            |            |          |   Reorder |           |      (excl. |
|    |                                    |            |            |          |           |           |   SetInput) |
|----+------------------------------------+------------+------------+----------+-----------+-----------+-------------|
|  0 | deploy_resnet50_int8_perf_sched_1k |     0.0426 |     0.0293 |   1.1042 |    0.0158 |    0.0739 |      1.2232 |
|  1 | deploy_resnet50_int8_simple_sched  |     0.0379 |     0.0293 |   1.5079 |    0.0141 |    0.0699 |      1.6211 |
+----+------------------------------------+------------+------------+----------+-----------+-----------+-------------+
```
