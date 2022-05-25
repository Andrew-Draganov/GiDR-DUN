import numpy as np

prefix = "outputs/data_size_timing/mnist"

print(["/rapids_umap", "/recreate_tsne_gpu", "/recreate_umap_gpu"])
for data_size in ["/1000_points", "/2000_points", "/4000_points", "/8000_points",
                  "/16000_points", "/32000_points", "/64000_points", "/128000_points"]:
    times = []
    for algorithm in ["/rapids_umap", "/recreate_tsne_gpu", "/recreate_umap_gpu"]:
        file = np.load(prefix + data_size + algorithm+ "/times.npy", allow_pickle=True)
        file = file[()]
        total_time = file["total_time"]
        times.append(total_time)
    print(data_size, times)
