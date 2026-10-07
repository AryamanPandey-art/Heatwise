import heatwise
df, summary, trace = heatwise.run_batch_simulation_benchmark(num_workloads=15)
print("Benchmark Summary:")
for k, v in summary.items():
    print(f"  {k}:")
    for kk, vv in v.items():
        print(f"    {kk}: {vv}")
