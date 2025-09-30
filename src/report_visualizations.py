import os
import pandas as pd
import matplotlib.pyplot as plt
import polars as pl
import geopandas as gpd
import seaborn as sns
from fastai.vision.all import *
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.image as mpimg

def create_data_availability_plot(lazy_df, datetime_col, start_date, end_date, temp_dir):
    """Create data availability plots that scale from hours to months of data"""
    def parse_date_string(date_str):
        # Standardize the date string format by appending ' 00:00:00' if no time is present
        # This occurs when the image is taken at 00:00
        if len(date_str.split(' ')) == 1:  # Only date part present
            date_str += ' 00:00:00'

        # Parse the standardized string into a datetime object
        date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        return date_obj
    
    # Parse the date strings into datetime objects
    start_date = parse_date_string(start_date)
    end_date = parse_date_string(end_date)

    print(f"type(start_date): {type(start_date)}; type(end_date): {type(end_date)}")
    print(f"Values: {start_date, end_date}")

    # First calculate the total duration in days
    duration_days = (start_date - end_date).days + 1  # +1 to count both start and end days

    # Create output directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)

    # Process the data at full resolution (10-minute intervals)
    full_resolution_data = (
        lazy_df
        .with_columns(
            (
                pl.col("date") + " " + pl.col("time")
            ).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M", strict=False).alias(datetime_col)
        )
        # Sort and group by dynamic time intervals (10 minutes)
        .sort(by=datetime_col)
        .group_by_dynamic(datetime_col, every="10m")
        .agg(pl.count().alias("count"))
        .collect()  # Collect the LazyFrame into a DataFrame after grouping
        .to_pandas()
    )

    # Create appropriate visualizations based on duration
    if duration_days <= 30:  # 1 week or less
        # Create hourly aggregated view for periods up to 1 month
        path = create_hourly_plot(full_resolution_data, datetime_col, start_date, end_date, temp_dir)
    elif duration_days <=90:
        # Create daily aggregated view for periods up to 3 months
        path = create_daily_plot(full_resolution_data, datetime_col, start_date, end_date, temp_dir)
    else:
        # For very long periods, create weekly summary
        path = create_weekly_plot(full_resolution_data, datetime_col, start_date, end_date, temp_dir)
    return path

def create_hourly_plot(data, datetime_col, start_date, end_date, temp_dir):
    """Create hourly aggregated plot for medium time periods"""
    # Aggregate data to hourly resolution
    hourly_data = (
        data
        .set_index(datetime_col)
        .resample('1h')['count']
        .sum()
        .reset_index()
    )

    # Generate full hourly time range
    time_range_pd = pd.date_range(start_date, end_date, freq='1h')
    full_time_range_df_pd = pd.DataFrame({'datetime': time_range_pd})

    # Merge with our data
    merged_data = pd.merge(
        full_time_range_df_pd,
        hourly_data,
        on='datetime',
        how='left'
    ).fillna({'count': 0})

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 2))

    # Calculate bar width (1 hour in days)
    interval_duration_days = 1 / 24
    colors = ['#004A6D' if count > 0 else 'grey' for count in merged_data['count']]

    ax.bar(merged_data['datetime'],
           height=[1]*len(merged_data),
           width=interval_duration_days,
           color=colors)

    # For 1 week to 1 month, use daily ticks
    tick_interval = 24  # Daily ticks
    date_format = '%a %d-%m'  # Weekday and date

    ax.set_xticks(merged_data['datetime'][::tick_interval])
    ax.set_xticklabels(merged_data['datetime'][::tick_interval].dt.strftime(date_format), rotation=45)

    # Add vertical grid lines at midnight each day
    for day in pd.date_range(start_date, end_date, freq='D'):
        ax.axvline(x=day, color='gray', linestyle='--', linewidth=0.5)

    ax.set_xlim(merged_data['datetime'].min(), merged_data['datetime'].max())
    #ax.set_xlabel('Time')
    ax.set_title('Data availability')
    ax.set_yticks([])
    ax.grid(which='major', axis='y', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()

    path = os.path.join(temp_dir, 'data_availability.png')
    plt.savefig(path)
    plt.close()
    
    return path

def create_daily_plot(data, datetime_col, start_date, end_date, temp_dir):
    """Create daily aggregated plot for longer time periods"""
    # Aggregate data to daily resolution
    daily_data = (
        data
        .set_index(datetime_col)
        .resample('1D')['count']
        .sum()
        .reset_index()
    )

    # Generate full daily time range
    time_range_pd = pd.date_range(start_date, end_date, freq='1D')
    full_time_range_df_pd = pd.DataFrame({'datetime': time_range_pd})

    # Merge with our data
    merged_data = pd.merge(
        full_time_range_df_pd,
        daily_data,
        on='datetime',
        how='left'
    ).fillna({'count': 0})

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 2))

    # Calculate bar width (1 day in days)
    interval_duration_days = 1
    colors = ['#004A6D' if count > 0 else 'grey' for count in merged_data['count']]

    ax.bar(merged_data['datetime'],
           height=[1]*len(merged_data),
           width=interval_duration_days,
           color=colors)

    # For monthly periods, use weekly ticks
    tick_interval = 7  # Weekly ticks
    date_format = '%b %d'  # Month and day

    ax.set_xticks(merged_data['datetime'][::tick_interval])
    ax.set_xticklabels(merged_data['datetime'][::tick_interval].dt.strftime(date_format), rotation=45)

    # Add vertical grid lines at start of each month
    for month_start in pd.date_range(start_date, end_date, freq='MS'):
        ax.axvline(x=month_start, color='gray', linestyle='--', linewidth=0.5)

    ax.set_xlim(merged_data['datetime'].min(), merged_data['datetime'].max())
    #ax.set_xlabel('Date')
    ax.set_title('Data Availability')
    ax.set_yticks([])
    ax.grid(which='major', axis='y', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()

    path = os.path.join(temp_dir, 'data_availability.png')
    plt.savefig(path)
    plt.close()

    return path

def create_weekly_plot(data, datetime_col, start_date, end_date, temp_dir):
    """Create weekly summary plot for very long time periods"""
    # Aggregate data to weekly resolution with coverage percentage
    # First calculate total possible measurements per week (assuming 10min intervals)
    # 7 days * 24 hours/day * 6 measurements/hour = 1008 possible measurements per week
    weekly_measurements = 7 * 24 * 6

    weekly_data = (
        data
        .set_index(datetime_col)
        .resample('1W-MON')['count']  # Week starting Monday
        .agg(['sum', 'count'])
        .reset_index()
    )
    weekly_data['coverage'] = weekly_data['sum'] / weekly_measurements

    # Generate full weekly time range (Mondays)
    time_range_pd = pd.date_range(start_date, end_date, freq='W-MON')
    full_time_range_df_pd = pd.DataFrame({'datetime': time_range_pd})

    # Merge with our data
    merged_data = pd.merge(
        full_time_range_df_pd,
        weekly_data,
        on='datetime',
        how='left'
    ).fillna({'sum': 0, 'count': 0, 'coverage': 0})

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 4))

    # Create a bar plot showing coverage percentage
    bars = ax.bar(merged_data['datetime'],
                 merged_data['coverage'] * 100,  # Convert to percentage
                 width=5,  # About 5 days in plot units for weekly bars
                 color=['#004A6D' if cov > 0 else 'grey' for cov in merged_data['coverage']])

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        if height > 0:  # Only label bars with data
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.0f}%',
                   ha='center', va='bottom', fontsize=8)

    # Format x-axis with monthly ticks
    monthly_ticks = pd.date_range(start_date, end_date, freq='MS')
    ax.set_xticks(monthly_ticks)
    ax.set_xticklabels(monthly_ticks.strftime('%b %Y'), rotation=45)

    # Add horizontal line at 100% coverage for reference
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=0.5)

    ax.set_xlim(merged_data['datetime'].min() - pd.Timedelta(days=3),
                merged_data['datetime'].max() + pd.Timedelta(days=3))
    ax.set_ylabel('Coverage Percentage')
    ax.set_title('Weekly Data Coverage')
    ax.set_ylim(0, 105)  # Coverage from 0 to 105% (bit above 100)
    ax.grid(which='major', axis='y', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()

    path = os.path.join(temp_dir, 'data_availability.png')
    plt.savefig(path)
    plt.close()
    
    return path

# Plot cruise path
def plot_cruise_path(cruise_path, cruise_fig_path, CRUISE_NAME):
    # Load the shapefile for the map background
    coastlines = gpd.read_file("data/eea_v_3035_100_k_coastline-poly_1995-2017_p_v03_r00")
    eez = gpd.read_file("data/EEZ_land_union_v4_202410")

    # Convert CRS from EPSG:4326 to EPSG:4258
    cruise_path = cruise_path.to_crs(epsg=4258)
    coastlines = coastlines.to_crs(epsg=4258)
    eez = eez.to_crs(epsg=4258)

    # Plot the map
    fig, ax = plt.subplots(figsize=(6, 8))

    # Plot EEZ data
    eez.plot(ax=ax, facecolor='none', edgecolor='gray',  linewidth=0.3, zorder=2)

    # Plot coastlines data with no fill and lowest zorder
    coastlines.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5,zorder=0)

    # Plot the cruise path with a lower zorder to ensure dots are on top
    cruise_path.plot(ax=ax, color='black', linewidth=2, label='Cruise Path', zorder=1)

    # Set plot extent
    minx, miny, maxx, maxy = cruise_path.total_bounds
    padding_degrees = 0.2  # In degrees
    minx_padded = minx - padding_degrees
    maxx_padded = maxx + padding_degrees
    miny_padded = miny - padding_degrees
    maxy_padded = maxy + padding_degrees

    ax.set_xlim(minx_padded, maxx_padded)
    ax.set_ylim(miny_padded, maxy_padded)

    # Format axes
    def format_lat(x, pos):
        return f"{x:.1f}°{'N' if x >= 0 else 'S'}"
    def format_lon(x, pos):
        return f"{x:.1f}°{'E' if x >= 0 else 'W'}"
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

    # Add grid lines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1.5)
    x_extent = maxx_padded - minx_padded # Calculate east-west extent
    lon_spacing = 2.0 if x_extent > 5 else 0.5 # Determine longitude grid spacing based on extent
    ax.xaxis.set_major_locator(MultipleLocator(base=lon_spacing))
    ax.yaxis.set_major_locator(MultipleLocator(base=lon_spacing))

    # Add text in the top-left corner 
    text_content = f"Cruise: {CRUISE_NAME}"
    ax.text(0.02, 0.98, text_content, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig(cruise_fig_path)
    print("[INFO] Finished cruise-path map")
    return fig

# Number of samples
def plot_image_count(stats_df, output_path):
    # Create a new column with processed class names
    stats_df['Processed_Class'] = stats_df['Class'].apply(lambda x: x.replace('_', ' ').replace('-', ' '))

    # Sort the DataFrame by the processed class names
    sorted_stats_df = stats_df.sort_values(by='Processed_Class', ascending=False)

    # Extract the sorted processed class names and counts
    sorted_processed_classes = sorted_stats_df['Processed_Class'].tolist()
    sorted_num_images = sorted_stats_df['# of Images'].tolist()

    # Create the plot
    plt.figure(figsize=(6, 8))
    plt.barh(sorted_processed_classes, sorted_num_images, color='skyblue')
    plt.xscale('log')
    plt.xlim(1, None)  # Start x-axis at 1 to avoid log(0)
    plt.xlabel('Number of Images')
    plt.ylabel('')
    # plt.title('Number of Images per Class')
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path)
    # plt.show()

# Confidence graph
def plot_confidence(class_df):
    print(f"[INFO] Started confidence plot for {class_df['pred_label'].first()}")
    class_df = class_df.to_pandas() # Convert Polars to Pandas

    # Calculate mean confidence for each of the 49 columns
    # Select all columns ending with '_conf' except 'pred_conf' and its duplicated version
    all_columns = class_df.columns.tolist()
    conf_columns = [col for col in all_columns if col.endswith('_conf') and col not in ['pred_conf', 'pred_conf_duplicated_0']]
    df_conf = class_df[conf_columns]

    mean_confidences = df_conf.mean()

    # Select five classes with the highest mean confidence, excluding the pred_id column
    pred_label = class_df['pred_label'].iloc[0]  # Assuming all pred_ids are the same as per your filter
    top_five_classes = mean_confidences.drop(f"{pred_label}_conf").nlargest(5).index

    # Create a horizontal violin plot
    fig, ax = plt.subplots(figsize=(6, 5))

    # Extract relevant data from the predicted class and top-5 related ones
    # And convert decimals to percentages
    data_to_plot = [class_df[idx] * 100 for idx in top_five_classes] + [class_df['pred_conf'] * 100]

    # Pretty-print the labels
    labels_top_five = []
    for class_conf in top_five_classes:
        base_name = class_conf.replace('_conf', '')
        labels_top_five.append(re.sub(r'[_\-]', '\n', base_name))

    pred_conf_label = re.sub(r'[_\-]', '\n', pred_label)
    full_labels = labels_top_five + [pred_conf_label] 
    
    # Create a list of colors, with a different color for the actual target
    colors = ['#004A6D'] * len(top_five_classes) + ['#FF5733']  # Using a different color for the actual target

    # Create violin plot
    parts = plt.violinplot(data_to_plot, vert=False, showmeans=True)

    # Set colors for each part of the violin plot
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor('black')

    plt.yticks(range(1, len(full_labels) + 1), full_labels)
    plt.title(f"{class_df['pred_label'].iloc[0]} vs top-5 related classes", fontsize=11)
    plt.xlabel('Confidence [%]')
    ax.grid(which='major', axis='y', linestyle='--', linewidth=0.5)

    plt.tight_layout()

    return fig

# Line-graph
def plot_density_graph(class_df, class_id, pred_labels):
    print(f"[INFO] Started line graph for {class_df['pred_label'].first()}")
    grouped_df = (
        class_df
        .sort(by='datetime')
        .group_by_dynamic("datetime", every="10m")
        .agg([
            pl.col("id").first(),
            pl.col("date").first(),
            pl.col("time").first(),
            pl.col("pred_conf").mean(),#.alias("pred_conf"),
            pl.col("pred_id").first(),
            pl.col("pred_label").first(),
            pl.col("density").mean()
        ])
    )
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
    #print("[INFO] Finished line plot")

    return fig

# Map
def plot_class_density_map(class_df, class_id, pred_labels, cruise_path, CRUISE_NAME):
    print(f"[INFO] Started density map for class #{class_id}")
    grouped_df = (
        class_df
        .sort(by='datetime')
        .group_by_dynamic("datetime", every="10m")
        .agg([
            pl.col("id").first(),
            pl.col("date").first(),
            pl.col("time").first(),
            pl.col("pred_conf").mean(),#.alias("pred_conf"),
            pl.col("pred_id").first(),
            pl.col("pred_label").first(),
            pl.col("density").mean(),
            pl.col("lat").first(),
            pl.col("lon").first()
        ])
    )

    # Convert to a GeoDataFrame
    df_pandas = grouped_df.to_pandas().set_index('datetime')

    # Check if there are any valid lat and lon values
    if df_pandas['lat'].isna().all() or df_pandas['lon'].isna().all():
        print("[WARNING] No valid lat-lon data found. Skipping plot.")
        return None

    # Convert the class-specific DataFrame to a GeoDataFrame
    gdf = (
        gpd.GeoDataFrame(
            df_pandas,
            geometry=gpd.points_from_xy(grouped_df['lon'], grouped_df['lat'], crs="EPSG:4326")
        )
    )

    # Load the shapefile for the map background
    coastlines = gpd.read_file("data/eea_v_3035_100_k_coastline-poly_1995-2017_p_v03_r00")
    eez = gpd.read_file("data/EEZ_land_union_v4_202410")

    # Convert CRS from EPSG:4326 to EPSG:4258
    coastlines = coastlines.to_crs(epsg=4258)
    eez = eez.to_crs(epsg=4258)
    gdf = gdf.to_crs(epsg=4258)
    cruise_path = cruise_path.to_crs(epsg=4258)

    # Plot the map
    fig, ax = plt.subplots(figsize=(6, 8))

    # Plot EEZ data
    eez.plot(ax=ax, facecolor='none', edgecolor='gray',  linewidth=0.3, zorder=2)

    # Plot coastlines data with no fill and lowest zorder
    coastlines.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5,zorder=0)

    # Plot the cruise path with a lower zorder to ensure dots are on top
    cruise_path.plot(ax=ax, color='black', linewidth=2, label='Cruise Path', zorder=1)

    # 1. Normalize the density values for color mapping
    min_density = 0 # Hard-code the minimum boundary to 0 N/L
    norm = plt.Normalize(min_density, gdf['density'].max())

    # 2. Create colormap - let's use a sequential color scheme from light to dark
    cmap = plt.colormaps['YlOrRd']  # Yellow to Orange to Red

    # 3. Scale marker sizes based on density (using square root to make sizes perceptually balanced)
    min_size = 20
    max_size = 140
    density_range = gdf['density'].max() - min_density
    gdf['marker_size'] = min_size + (max_size - min_size) * (gdf['density'] - min_density) / density_range

    # Handle special case where density_range is 0 (all densities equal)
    if density_range == 0:
        gdf['marker_size'] = min_size

    # Plot each point with color and size based on density
    sc = ax.scatter(
        gdf.geometry.x,
        gdf.geometry.y,
        c=gdf['density'],
        cmap=cmap,
        norm=norm,
        s=gdf['marker_size'],
        edgecolor='black',
        alpha=1.0, # No transparency
        zorder=3
    )

    # Add a colorbar to show the density gradient
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5)
    cbar.set_label('Density (N/L)', rotation=270, labelpad=15)

    # Set plot extent
    minx, miny, maxx, maxy = cruise_path.total_bounds
    padding_degrees = 0.2  # In degrees
    minx_padded = minx - padding_degrees
    maxx_padded = maxx + padding_degrees
    miny_padded = miny - padding_degrees
    maxy_padded = maxy + padding_degrees

    ax.set_xlim(minx_padded, maxx_padded)
    ax.set_ylim(miny_padded, maxy_padded)

    # Format axes
    def format_lat(x, pos):
        return f"{x:.1f}°{'N' if x >= 0 else 'S'}"
    def format_lon(x, pos):
        return f"{x:.1f}°{'E' if x >= 0 else 'W'}"
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

    # Add grid lines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1.5)
    x_extent = maxx_padded - minx_padded # Calculate east-west extent
    lon_spacing = 2.0 if x_extent > 5 else 0.5 # Determine longitude grid spacing based on extent
    ax.xaxis.set_major_locator(MultipleLocator(base=lon_spacing))
    ax.yaxis.set_major_locator(MultipleLocator(base=lon_spacing))

    # Add text in the top-left corner (as before)
    pred_label = pred_labels[class_id]
    text_content = f"Cruise: {CRUISE_NAME}\nSpecies: {pred_label}"
    ax.text(0.02, 0.98, text_content, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

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

    # Sort the sampled images by confidence in descending order
    sampled_df = sampled_df.sort("pred_conf", descending=True)

    # Extract the file paths and other needed data from the sampled DataFrame
    filenames = (sampled_df['id']
                .str.replace_all(r"\\", "/")  # Normalize slashes
                .str.split("/")               # Split by slash
                .list.get(-1)                 # Take last part (filename)
                .to_list())
    pred_confs = sampled_df['pred_conf'].to_list()
    tar_files = sampled_df['tar_file'].to_list()

    # Create a figure to plot the images
    n_cols = 8
    n_rows = (len(filenames) + n_cols - 1) // n_cols  # Calculate the number of rows needed
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6, 7))
    
    # Ensure axes is always treated as an iterable, even if only 1 sample is available
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    # Plot each image with its corresponding confidence value
    for ax, (filename, pred_conf, tar_path) in zip(axes, zip(filenames, pred_confs, tar_files)):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Open the tar file
            with tarfile.open(tar_path, 'r') as tar:
                for path in (f"RawImages/{filename}", f"RawImages\\{filename}"): # Try both Windows and Unix style
                    try:
                        # Extract .tif file
                        tar.extract(path, path=temp_dir)
                        extracted_path = os.path.join(temp_dir, path.replace('\\', '/'))
                        if os.path.exists(extracted_path):
                            img = mpimg.imread(extracted_path)
                            ax.imshow(img)
                            ax.axis('off')
                            ax.set_title(f"Conf: {pred_conf:.2f}", fontsize=6)
                            file_found = True
                            break
                    except KeyError:
                        continue
                ax.set_title(f"Conf: {pred_conf:.2f}", fontsize=6)  # Set the title with the confidence value

    # Hide any unused subplots
    for ax in axes[len(filenames):]:
        ax.axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    return fig
