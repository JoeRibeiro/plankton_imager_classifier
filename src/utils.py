import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from fastai.vision.all import *
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.image as mpimg
import polars as pl
import gc
from memory_profiler import profile


""" UNUSED SCRIPT FOR NOW. IMPLEMENT LATER ON WHEN DEBUGGING IS FINISHED"""

# Confidence graph
def plot_confidence(class_df):
    print(f"[INFO] Started confidence plot #{class_df['pred_label'].first()}")
    class_df = class_df.to_pandas() # Convert Polars to Pandas

    # Calculate mean confidence for each of the 49 columns
    mean_confidences = class_df.iloc[:, 2:51].mean()

    # Select five classes with the highest mean confidence, excluding the pred_id column
    pred_id = class_df['pred_id'].iloc[0]  # Assuming all pred_ids are the same as per your filter
    top_five_classes = mean_confidences.drop(str(pred_id)).nlargest(5).index

    # Create a horizontal violin plot
    fig, ax = plt.subplots(figsize=(6, 5))
    data_to_plot = [class_df[str(idx)] for idx in top_five_classes] + [class_df['conf']]
    labels = [pred_labels[int(idx)].replace('_', '\n') for idx in top_five_classes] + [pred_labels[pred_id].replace('_', '\n')]

    # Create a list of colors, with a different color for the actual target
    colors = ['#004A6D'] * len(top_five_classes) + ['#FF5733']  # Using a different color for the actual target

    # Create violin plot
    parts = plt.violinplot(data_to_plot, vert=False, showmeans=True)

    # Set colors for each part of the violin plot
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor('black')

    plt.yticks(range(1, len(labels) + 1), labels)
    plt.title(f'{class_df['pred_label'].iloc[0]} vs top-5 related classes', fontsize=11)
    plt.xlabel('Confidence [-]')
    ax.grid(which='major', axis='y', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    #plt.show()
    print("[INFO] Finished confidence plot")

    return fig

# Line-graph
def plot_density_graph(grouped_df, class_id):
    print(f"[INFO] Started line graph for class #{class_id}")
    # Filter out None values from unique_dates
    grouped_df = grouped_df.to_pandas()
    unique_dates = [date for date in grouped_df['date'].unique() if date is not None]

    # Plotting code
    n_cols = 4
    n_rows = (len(unique_dates) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), sharey=True)
    axes = axes.flatten()
    pred_label = pred_labels[class_id] # Return pred label
    fig.suptitle(f"Class: {pred_label}", fontsize=11)

    for idx, (ax, date) in enumerate(zip(axes, unique_dates)):
        full_time_range = pd.date_range(f"{date} 00:00:00", f"{date} 23:59:59", freq='10min')
        date_df = grouped_df[grouped_df['date'] == date]

        # Create a DataFrame with the full time range
        full_df = pd.DataFrame({'datetime': full_time_range})
        full_df['density'] = full_df['datetime'].map(dict(zip(date_df['datetime'], date_df['density']))).fillna(0)

        # Plot using Seaborn lineplot with specified line color
        sns.lineplot(x=full_df['datetime'], y='density', data=full_df, ax=ax, color='#004A6D')

        # Fill the area under the line plot
        ax.fill_between(full_df['datetime'], full_df['density'], color='#004A6D', alpha=0.3)

        # Set x-ticks every two hours
        ax.set_xticks(full_time_range[::12])  # Every two hours since 10min intervals
        ax.set_xticklabels(full_time_range[::12].strftime('%H:00'), rotation=45)
        ax.set_title(f"{date}")

        # Only set y-label for the leftmost plots
        if idx % n_cols == 0:
            ax.set_ylabel('Density [N/L]')
        else:
            ax.set_ylabel('')

        # Only set x-ticks for the bottom plots
        if idx >= len(axes) - n_cols:
            ax.set_xticklabels(full_time_range[::12].strftime('%H:00'), rotation=45)
        else:
            ax.set_xticklabels([])

        # Explicitly set x-axis label to an empty string
        ax.set_xlabel('')

        # Force the axis to save on white space
        ax.set_ylim(bottom=0, top=grouped_df['density'].max())
        ax.set_xlim(full_time_range[0], full_time_range[-1])

        # Add vertical lines at every two hours
        for hour in full_time_range[::12]:
            ax.axvline(x=hour, color='gray', linestyle='--', linewidth=0.5)

        ax.grid(which='major', axis='y', linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

    for j in range(len(unique_dates), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()  # Without any rectifications
    print("[INFO] Finished line plot")

    return fig

# Map
def plot_class_density_map(class_df, class_id, cruise_path, OSPAR, CRUISE_NAME):
    print(f"[INFO] Started density map for class #{class_id}")
    # TODO: Remove these lines
    path_0000 = r"C:\Users\dalen024\Documents\MONS_data\2024_MONS_Tridens_january_UNTARRED_TEST\2024-01-18\untarred_0000\RawImages\pia7.2024-01-18.0000+N00000269.tif"
    path_0010 = r"C:\Users\dalen024\Documents\MONS_data\2024_MONS_Tridens_january_UNTARRED_TEST\2024-01-18\untarred_0010\RawImages\pia7.2024-01-18.0010+N00000269.tif"
    path_0020 = r"C:\Users\dalen024\Documents\MONS_data\2024_MONS_Tridens_january_UNTARRED_TEST\2024-01-18\untarred_0020\RawImages\pia7.2024-01-18.0020+N00000269.tif"
    path_0030 = r"C:\Users\dalen024\Documents\MONS_data\2024_MONS_Tridens_january_UNTARRED_TEST\2024-01-18\untarred_0030\RawImages\pia7.2024-01-18.0030+N00000269.tif"
    path_0000_bg = r"C:\Users\dalen024\Documents\MONS_data\2024_MONS_Tridens_january_UNTARRED_TEST\2024-01-18\untarred_0000\Background.tif"
    path_0010_bg = r"C:\Users\dalen024\Documents\MONS_data\2024_MONS_Tridens_january_UNTARRED_TEST\2024-01-18\untarred_0010\Background.tif"
    path_0020_bg = r"C:\Users\dalen024\Documents\MONS_data\2024_MONS_Tridens_january_UNTARRED_TEST\2024-01-18\untarred_0020\Background.tif"
    path_0030_bg = r"C:\Users\dalen024\Documents\MONS_data\2024_MONS_Tridens_january_UNTARRED_TEST\2024-01-18\untarred_0030\Background.tif"

    # Create a list of paths alternating among the defined paths
    paths = [
        path_0000 if i % 8 == 0 else
        path_0010 if i % 8 == 1 else
        path_0020 if i % 8 == 2 else
        path_0030 if i % 8 == 3 else
        path_0000_bg if i % 8 == 4 else
        path_0010_bg if i % 8 == 5 else
        path_0020_bg if i % 8 == 6 else
        path_0030_bg
        for i in range(len(class_df))
    ]

    # Assign the paths to the 'id' column
    # TODO: Remove all these lines
    class_df = class_df.with_columns(pl.Series("id", paths))

    # Create a new column for the latitude-longitude EXIF metadata
    class_df = class_df.with_columns(
        pl.col("id")
        .map_elements(
            lambda x: dict(zip(("lat", "lon"), get_geographic_data(x))),
            return_dtype=pl.Struct([pl.Field("lat", pl.Float64), pl.Field("lon", pl.Float64)]) # Specify dtype to silence Polars warning
        )
        .alias("coordinates")
    ).unnest("coordinates")  # Unnest into lat-lon columns; We need separate columns for GeoPandas to get geometry

    # Convert the class-specific DataFrame to a GeoDataFrame
    gdf = (
        gpd.GeoDataFrame(
            class_df.to_pandas().set_index('datetime'), # Get Pandas DataFrame
            geometry=gpd.points_from_xy(class_df['lon'], class_df['lat'], crs="EPSG:4326")
        )
    )

    # Load the shapefile for the map background
    world = gpd.read_file(OSPAR)

    # Convert CRS from EPSG:4326 to EPSG:4258
    world = world.to_crs(epsg=4258)
    gdf = gdf.to_crs(epsg=4258)

    # Define density classes and corresponding marker sizes for the map
    density_classes = [0.0, 0.2, 0.4, 0.6, float('inf')]
    map_marker_sizes = [20, 60, 100, 140]
    legend_marker_sizes = [4, 6, 8, 10]

    # Define colors for each density class
    density_colors = ["#f6aa1c", "#bc3908", "#941b0c", "#621708"]

    # Create a new column in the GeoDataFrame to categorize density
    gdf['density_class'] = pd.cut(gdf['density'], bins=density_classes, labels=range(len(density_colors)))

    # Check for zero samples and assign a gray dot
    gdf['density_class'] = gdf['density_class'].cat.add_categories([-1])  # Use -1 for zero density
    gdf.loc[gdf['density'] <= 0.01, 'density_class'] = -1 # Assign values below 0.01 N/L as absence

    # Filter out data points with no marker size
    gdf = gdf.dropna(subset=['density_class'])

    # Plot the map
    fig, ax = plt.subplots(figsize=(6, 7))

    # Define colors for each category
    category_colors = {
        'Coastal': '#BFEFFF',
        'Shelf': '#9AC0CD',
        'River plumes': '#D2B48C'
    }

    # Plot each OSPAR category with its respective color
    for category, color in category_colors.items():
        world[world['Category'] == category].plot(ax=ax, color=color, edgecolor='black')

    # Plot the cruise path with a lower zorder to ensure dots are on top
    cruise_path.plot(ax=ax, color='black', linewidth=2, label='Cruise Path', zorder=1)

    # Plot the class data on top with density classes as different marker sizes and colors
    for idx, row in gdf.iterrows():
        if row['density_class'] == -1:  # Zero density
            ax.scatter(row['geometry'].x, row['geometry'].y, marker='o', color='gray', s=map_marker_sizes[0], edgecolor='black', alpha=1.0, zorder=3)
        else:
            # Determine the color based on the density class
            color = density_colors[int(row['density_class'])]
            ax.scatter(row['geometry'].x, row['geometry'].y, marker='o', color=color, s=map_marker_sizes[int(row['density_class'])], edgecolor='black', alpha=1.0, zorder=3)

    # Set the plot extent based on the bounds of the cruise path data
    minx, miny, maxx, maxy = cruise_path.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    # Format the x and y ticks to display degrees with N, S, E, W
    def format_lat(x, pos):
        return f"{x:.1f}°{'N' if x >= 0 else 'S'}"

    def format_lon(x, pos):
        return f"{x:.1f}°{'E' if x >= 0 else 'W'}"

    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

    # Add grid lines at every half degree, below the density dots but above the background
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1.5)
    ax.xaxis.set_major_locator(MultipleLocator(base=0.5))
    ax.yaxis.set_major_locator(MultipleLocator(base=0.5))

    # Add text in the top-left corner
    pred_label = pred_labels[class_id]  # Return pred label
    text_content = f"Cruise: {CRUISE_NAME}\nSpecies: {pred_label}"
    ax.text(0.02, 0.98, text_content, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Create legend entries for the density markers with smaller sizes
    legend_patches_markers = [
        Line2D([0], [0], marker='o', color='w', label='Absence', markerfacecolor='gray', markersize=legend_marker_sizes[0], markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', label='0.01-0.2 N/L', markerfacecolor=density_colors[0], markersize=legend_marker_sizes[0], markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', label='0.2-0.4 N/L', markerfacecolor=density_colors[1], markersize=legend_marker_sizes[1], markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', label='0.4-0.6 N/L', markerfacecolor=density_colors[2], markersize=legend_marker_sizes[2], markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', label='>0.6 N/L', markerfacecolor=density_colors[3], markersize=legend_marker_sizes[3], markeredgecolor='black'),
    ]

    # Create a legend for the markers
    plt.legend(handles=legend_patches_markers, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=3)
    plt.tight_layout()

    print("[INFO] Finished density map")
    return fig

def plot_random_images(class_df, num_images=80):
    print(f"[INFO] Started sampling random images for {class_df['pred_label'].first()}")
    # Randomly sample rows from the DataFrame
    try:
        # Normally you should have more images predicted than num_samples
        sampled_df = class_df.sample(n=num_images, seed=42)
    except:
        # If you have less, we use with_replacement to handle this case
        # But this should not occur, unless an extremely small dataset is used
        print(f"[WARNING] Only {class_df.height} images found for class {class_df['pred_label'].first()}. {num_images} images required")
        sampled_df = class_df.sample(n=class_df.height, seed=42) # No sample, just take all available images (<80img)

    # Extract the file paths from the sampled DataFrame
    sampled_paths = sampled_df['id'].to_list()

    # Create a figure to plot the images
    n_cols = 8
    n_rows = (len(sampled_paths) + n_cols - 1) // n_cols  # Calculate the number of rows needed
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6, 7))
    
    # Ensure axes is always treated as an iterable, even if only 1 sample is available
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    # Plot each image with its corresponding confidence value
    for ax, (path, conf) in zip(axes, zip(sampled_paths, sampled_df['conf'])):
        # TODO: Remove path
        path = r"C:\Users\dalen024\Documents\MONS_data\2024_MONS_Tridens_january_UNTARRED_TEST\2024-01-16\untarred_0000\RawImages\pia7.2024-01-16.0000+N00000000.tif"
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.axis('off')  # Hide axes
        ax.set_title(f"Conf: {conf:.2f}", fontsize=6)  # Set the title with the confidence value

    # Hide any unused subplots
    for ax in axes[len(sampled_paths):]:
        ax.axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    #plt.show()
    print("[INFO] Finished plotting random images")
    return fig
