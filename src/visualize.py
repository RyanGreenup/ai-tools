#!/usr/bin/env python3

from bokeh.layouts import gridplot
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.models import LinearColorMapper
from bokeh.plotting import figure, show, output_file
from embeddings.build import build_embeddings, initialize_chromadb
from enum import Enum
from matplotlib import colors as plt_colors
from numpy import ndarray
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import Iterable
from markdown import markdown
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import os
import sys


def plot_with_hover(
    x: Iterable[float],
    y: Iterable[float],
    c: Iterable[float],
    names: Iterable[str],
):
    """
    Plot with hover annotations

    Example:
    plot_with_hover(
        x: ndarray = np.random.rand(15),
        y: ndarray = np.random.rand(15),
        c: ndarray = np.random.randint(1, 5, size=15),
        names: ndarray = np.array(list("ABCDEFGHIJKLMNO")),
    )
    """
    norm = plt_colors.Normalize(1, 4)
    cmap = plt.get_cmap("Spectral")

    fig, ax = plt.subplots()
    sc = plt.scatter(x, y, c=c, s=100, cmap=cmap, norm=norm)

    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(20, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
        wrap=True,
    )
    annot.set_visible(False)

    def update_annot(ind: dict[str, ndarray]):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        # The index can be helpful for debugging (It's a list in case multiple are highlighted)
        # indices = [str(i) for i in ind["ind"]]
        # Use join for a safer pop, because multiple indices can be selected at once
        text = " ".join([names[n] for n in ind["ind"]])
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    n_docs = len(x)
    plt.title(f"Visualisation of Documents {n_docs=} in Semantic Space")
    fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.show()


def bokeh_plot(x,
               y,
               clusters,
               chunks,
               embed_model: str | None = None,
               dim_reducer: str | None = None
               ):
    # Define tools and tooltips
    TOOLS = "crosshair,pan,wheel_zoom,box_zoom,reset,hover,save"
    TOOLTIPS = """
    <div style="width:600px;">
    @chunks
    </div>
    """

    # Create the dataframe
    color_mapper = LinearColorMapper(
        palette="Magma256", low=min(clusters), high=max(clusters)
    )
    source = {
        "x": x,
        "y": y,
        "chunks": [markdown(c) for c in chunks],
        "clusters": clusters,
    }

    # Create the figure
    title = "Semantic Space"
    if embed_model:
        title = f"Embedding Space of {embed_model}"
    else:
        title = "Semantic Space"
    if dim_reducer:
        title = f"{title} via {dim_reducer}"
    p = figure(
        width=1500, height=1000, tooltips=TOOLTIPS, tools=TOOLS, title=title
    )

    # Create a Scatter plot
    p.scatter(
        x="x",
        y="y",
        size=20,
        color={"field": "clusters", "transform": color_mapper},
        alpha=0.5,
        source=source,
        legend_group="clusters",
    )

    # show the results
    output_file("/tmp/semantic_space_plot.html")
    show(p)


class DimensionReduction(Enum):
    PCA = "PCA"
    UMAP = "UMAP"
    TSNE = "TSNE"


def dim_reduction(
    embeddings: Iterable[Iterable[float]], method: DimensionReduction
) -> tuple[ndarray, ndarray]:
    """
    Squashings the embeddings into 2D space
    """
    match method:
        case DimensionReduction.PCA:
            # Perform pca over the embeddings
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(embeddings)
            x, y = pca_result[:, 0], pca_result[:, 1]
            return x, y
        case DimensionReduction.UMAP:
            # Lazy import
            from umap.umap_ import UMAP  # umap-learn

            # Don't set random state for parallelism
            reducer = UMAP(n_components=2)
            umap_result = reducer.fit_transform(embeddings)
            x, y = umap_result[:, 0], umap_result[:, 1]
            return x, y
        case DimensionReduction.TSNE:
                from sklearn.manifold import TSNE
                tsne = TSNE(n_components=2)
                tsne_result = tsne.fit_transform(embeddings)
                x, y = tsne_result[:, 0], tsne_result[:, 1]
                return x, y
        case _:
            raise ValueError("Invalid Dimension Reduction Method")


def cluster(embeddings: ndarray, n_clusters: int) -> ndarray:
    """
    Clusters the embeddings into n_clusters
    """
    kmeans = KMeans(n_clusters=n_clusters)
    return kmeans.fit_predict(embeddings)


def vis(db_location: str,
        notes_dir: Path,
        model_name: str,
        dim_reducer: DimensionReduction,
        ollama_host: str,
        ):
    # TODO scale size by pagerank

    # Get the embeddings and Metadata
    if not os.path.exists(db_location):
        print(
            f"Database not found at {db_location}, Building a new one", file=sys.stderr
        )
        collection = build_embeddings(db_location, notes_dir, model_name, ollama_host)
    else:
        collection = initialize_chromadb(db_location)

    db = collection.get(include=["embeddings", "metadatas", "documents"])
    ids = db["ids"]
    embeddings = db["embeddings"]
    paths = [d["path"] for d in db["metadatas"]]
    chunks = db["documents"]
    title_paths = [
        str(os.path.basename(os.path.splitext(p)[0]))
        .title()
        .replace("_", "/")
        .replace("-", " ")
        for p in paths
    ]
    chunks = [f"# {p}\n{c}" for p, c in zip(title_paths, chunks)]

    # Perform KNN clustering
    clusters = cluster(embeddings, 7)

    # Squash
    x, y = dim_reduction(embeddings, dim_reducer)

    #  plot_with_hover(x, y, clusters, chunks)
    bokeh_plot(x, y, clusters, chunks, model_name, dim_reducer.value)
