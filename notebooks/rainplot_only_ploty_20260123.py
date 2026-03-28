import os
from argparse import ArgumentParser

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# --- 1. RainCloud Helper Function ---
def add_raincloud_trace(fig, df, row, col, palette):
    nodes = ["root", "1", "2", "3", "4"]
    for i, node in enumerate(nodes):
        node_df = df[df["node_name"] == node].copy()

        # Clip at 0.1 for log stability; labeling this as "0" in the axis
        node_df["percentage_log"] = node_df["percentage"].clip(lower=0.1)
        color = palette[i % len(palette)]

        # 1. The Cloud (Half-Violin)
        fig.add_trace(
            go.Violin(
                x=node_df["node_name"],
                y=node_df["percentage_log"],
                name=node,
                side="positive",
                line_color=color,
                fillcolor=color,
                opacity=0.4,
                width=1.0,  # Reduced width to prevent overlapping
                spanmode="hard",
                points=False,
                showlegend=False,
                meanline_visible=False,
                scalemode="count",
            ),
            row=row,
            col=col,
        )

        # 2. The Rain (Strip Plot)
        fig.add_trace(
            go.Box(
                x=node_df["node_name"],
                y=node_df["percentage_log"],
                name=node,
                boxpoints="all",
                jitter=0.3,  # Reduced jitter
                pointpos=-1.0,  # Adjusted position
                marker=dict(size=3, color=color, opacity=0.6),
                fillcolor="rgba(0,0,0,0)",
                line_color="rgba(0,0,0,0)",
                showlegend=False,
            ),
            row=row,
            col=col,
        )


def load_exp_data(search_dir: str, gen: int) -> pd.DataFrame:
    plot_info = []
    data_path = f"{search_dir}/gen_{gen}/"

    if os.path.exists(data_path):
        for root, sub_dir, files in os.walk(data_path):
            for file in files:
                if file.endswith(".csv"):
                    try:
                        src_file = os.path.join(root, file)
                        dim = int(file[3])
                        obj = int(file[9])
                        tree = file[16:].strip(".csv")
                        df = pd.read_csv(src_file, index_col=0).fillna(0)
                        df["dimension"] = dim
                        df["n_objectives"] = obj
                        df["tree"] = tree
                        plot_info.append(df)
                    except:
                        continue

    new_df = pd.concat(plot_info, ignore_index=True) if plot_info else pd.DataFrame()
    return new_df


# --- 3. Plotting Execution ---
def plot_rainclouds(data_df: pd.DataFrame, output_dir: str, gen: int):
    dim_n_obj_combs = [(5, 2), (4, 3), (3, 4), (2, 5)]
    solvers = ["GDE3", "NSGAII", "IBEA", "MOEAD"]
    trees = ["breadth", "depth"]
    colors = ["#8fdea0", "#ec80b4", "#eed5a4", "#9a97dc", "#d7e7f5"]
    output_dir = f"{output_dir}/gen_{gen}"
    os.makedirs(output_dir, exist_ok=True)
    for solver in solvers:
        for tree in trees:
            filtered_df = data_df[(data_df["solver"] == solver) & (data_df["tree"] == tree)]
            if filtered_df.empty:
                continue

            # Increased horizontal_spacing to prevent overlapping
            fig = make_subplots(
                rows=1, cols=4, subplot_titles=[f"N objectives = {obj}" for _, obj in dim_n_obj_combs], horizontal_spacing=0.08
            )

            for idx, (dim, n_objective) in enumerate(dim_n_obj_combs):
                sub_df = filtered_df[filtered_df["n_objectives"] == n_objective]
                if sub_df.empty:
                    continue

                plot_data = []
                for _, row_data in sub_df.iterrows():
                    for node_key in ["root", "node_1", "node_2", "node_3", "node_4"]:
                        name = node_key.replace("node_", "")
                        plot_data.append({"node_name": name, "percentage": row_data[node_key]})

                add_raincloud_trace(fig, pd.DataFrame(plot_data), 1, idx + 1, colors)

            solver_display = {"MOEAD": "MOEA/D", "NSGAII": "NSGA-II"}.get(solver, solver)

            # --- FIXED 4:1 Layout ---
            fig.update_layout(
                title={
                    "text": f"Distribution of {solver_display} Sampling Points ({tree} tree)",
                    "y": 0.92,  # Closer to subplots
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top",
                    "font": {"size": 22},
                },
                template="plotly_white",
                height=400,  # Set height
                width=1600,  # 1600 / 400 = Exactly 4:1
                violinmode="overlay",
                margin=dict(t=100, b=60, l=80, r=40),
                autosize=False,
            )

            # Axis configurations
            fig.update_yaxes(
                title_text="Percentage (%)",
                title_font={"size": 14},
                type="log",
                range=[-1.05, 2.1],
                tickvals=[0.1, 1, 10, 100],
                ticktext=["0", "1", "10", "100"],
                gridcolor="rgba(0,0,0,0.1)",
                zeroline=False,
            )

            fig.update_xaxes(
                title_text="Basin ID",
                title_font={"size": 14},
                tickmode="array",
                tickvals=["root", "1", "2", "3", "4"],
                ticktext=["root", "1", "2", "3", "4"],
                tickangle=0,
                tickfont=dict(size=12),
            )

            # Explicitly pass width/height to the export engine
            fig.write_image(f"{output_dir}/{solver}_{tree}.png", width=1600, height=400, scale=2)


def main():
    parser = ArgumentParser()
    parser.add_argument("--search_dir", type=str, default="stats_output")
    parser.add_argument("--gen", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="figures")
    args = parser.parse_args()
    search_dir = args.search_dir
    gen = args.gen
    output_dir = args.output_dir

    data_df = load_exp_data(search_dir, gen)
    if data_df.empty:
        print("No data found for the specified generation.")
        return

    plot_rainclouds(data_df, output_dir, gen)


if __name__ == "__main__":
    main()
