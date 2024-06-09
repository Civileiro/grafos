from main import perform_module, Part1
from tqdm import tqdm

gdir, gundir = perform_module(
    Part1,
    perform_inputs={"dataset_path": "./netflix_amazon_disney_titles.csv"},
)

for node in tqdm(list(gundir.vertices())):
    gundir.dijkstra(node)
