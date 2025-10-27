"""
Problem 2: Cluster Usage Analysis

This script analyzes cluster usage patterns to understand which clusters are
most heavily used over time. It extracts cluster IDs, application IDs, and
application start/end times to create a time-series dataset suitable for
visualization with Seaborn.

IMPORTANT: This problem takes approximately 10-20 minutes to run on the cluster.

Key Questions to Answer:
- How many unique clusters are in the dataset?
- How many applications ran on each cluster?
- Which clusters are most heavily used?
- What is the timeline of application execution across clusters?

**Expected output 1 (timeline):** 
```
cluster_id,application_id,app_number,start_time,end_time
1485248649253,application_1485248649253_0001,0001,2017-03-29 10:04:41,2017-03-29 10:15:23
1485248649253,application_1485248649253_0002,0002,2017-03-29 10:16:12,2017-03-29 10:28:45
1448006111297,application_1448006111297_0137,0137,2015-11-20 14:23:11,2015-11-20 14:35:22
...
```

**Expected output 2 (cluster summary):**
```
cluster_id,num_applications,cluster_first_app,cluster_last_app
1485248649253,181,2017-03-29 10:04:41,2017-03-29 18:42:15
1472621869829,8,2016-08-30 12:15:30,2016-08-30 16:22:10
...
```

**Expected output 3 (stats):**
```
Total unique clusters: 6
Total applications: 194
Average applications per cluster: 32.33

Most heavily used clusters:
  Cluster 1485248649253: 181 applications
  Cluster 1472621869829: 8 applications
  ...
```

**Visualization Output:**
The script automatically generates two separate visualizations:

1. **Bar Chart** (`problem2_bar_chart.png`):
   - Number of applications per cluster
   - Value labels displayed on top of each bar
   - Color-coded by cluster ID

2. **Density Plot** (`problem2_density_plot.png`):
   - Shows job duration distribution for the **largest cluster** (cluster with most applications)
   - Histogram with KDE overlay
   - **Log scale** on x-axis to handle skewed duration data
   - Sample count (n=X) displayed in title
"""

import argparse
import sys
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    regexp_extract,
    first,
    last,
    count,
    input_file_name,
    collect_list,
    min as spark_min,
    max as spark_max,
    udf,
    concat_ws,
)
from pyspark.sql.types import StringType
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# seaborn style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def parse_cluster_and_app_id(file_path):
    """
    Extract cluster_id and application_id from file path.
    
    Path format: application_<cluster_id>_<app_number>/container_*.log
    Example: application_1485248649253_0052/container_1485248649253_0052_01_000001.log
    Returns: (cluster_id, app_number, full_application_id)
    """

    import re
    match = re.search(r'application_(\d+)_(\d+)', file_path)
    if match:
        cluster_id = match.group(1)
        app_number = match.group(2)
        full_app_id = f"application_{cluster_id}_{app_number}"
        return cluster_id, app_number, full_app_id
    return None, None, None


def timestamp_to_datetime(timestamp_str):
    """Convert YY/MM/DD HH:MM:SS to datetime object."""
    try:
        return datetime.strptime(timestamp_str, '%y/%m/%d %H:%M:%S')
    except:
        return None


def main():
    parser = argparse.ArgumentParser(description='Analyze cluster usage patterns')
    parser.add_argument(
        'master_url',
        nargs='?',
        type=str,
        help='Spark master URL (e.g., spark://master:7077 or local[*])',
    )
    parser.add_argument(
        '--net-id',
        type=str,
        help='Your net ID for S3 bucket access',
    )
    parser.add_argument(
        '--local',
        action='store_true',
        help='Run locally using sample data instead of cluster',
    )
    parser.add_argument(
        '--skip-spark',
        action='store_true',
        help='Skip Spark processing and regenerate visualizations from existing CSVs',
    )
    
    args = parser.parse_args()
    
    # Determine data source
    if args.local:
        data_path = "data/sample/"
        output_prefix = "problem2"
        print("Running in LOCAL mode using sample data...")
    elif args.master_url:
        if not args.net_id:
            print("Error: --net-id is required when running on cluster")
            sys.exit(1)
        data_path = f"s3a://{args.net_id}-assignment-spark-cluster-logs/data/"
        output_prefix = "problem2"
        print(f"Running on cluster using S3 data: {data_path}")
    elif args.skip_spark:
        # Just regenerate visualizations from existing CSVs
        output_prefix = "problem2"
    else:
        print("Error: Either provide master_url, use --local flag, or use --skip-spark")
        sys.exit(1)
    
    if args.skip_spark:
        print("Skipping Spark processing. Regenerating visualizations from existing CSVs...")
    else:
        # Initialize Spark session
        builder = (
            SparkSession.builder
            .appName("Cluster Usage Analysis")
        )
        
        if args.master_url and not args.local:
            builder = (
                builder
                .master(args.master_url)
                .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
                .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.InstanceProfileCredentialsProvider")
                .config("spark.sql.adaptive.enabled", "true")
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
                .config("spark.executor.memory", "4g")
                .config("spark.driver.memory", "2g")
            )
        
        spark = builder.getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        
        try:
            print("Starting cluster usage analysis...")
            print("This may take 10-20 minutes to complete...")
            
            # Read ApplicationMaster logs only (container_*_01_000001.log)
            print(f"Reading ApplicationMaster logs from: {data_path}")
            logs_df = spark.read.text(f"{data_path}*/container_*_01_000001.log")
            
            # Extract file paths and timestamps
            enriched_logs = logs_df.withColumn(
                'file_path',
                input_file_name()
            ).withColumn(
                'timestamp',
                regexp_extract('value', r'^(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})', 1)
            ).filter(
                col('timestamp') != ''
            )
            
            # Extract cluster and application IDs from file paths
            print("Extracting cluster and application IDs...")
            
            # Define UDF to extract cluster_id
            def extract_cluster_id(path):
                import re
                match = re.search(r'application_(\d+)_(\d+)', path)
                return match.group(1) if match else None
            
            def extract_app_number(path):
                import re
                match = re.search(r'application_(\d+)_(\d+)', path)
                return match.group(2) if match else None
            
            extract_cluster_id_udf = udf(extract_cluster_id, StringType())
            extract_app_number_udf = udf(extract_app_number, StringType())
            
            enriched_logs = enriched_logs.withColumn(
                'cluster_id',
                extract_cluster_id_udf('file_path')
            ).withColumn(
                'app_number',
                extract_app_number_udf('file_path')
            )
            
            # Filter out null values
            enriched_logs = enriched_logs.filter(
                (col('cluster_id').isNotNull()) & 
                (col('app_number').isNotNull())
            )
            
            # Find first and last timestamps for each application
            print("Calculating application start and end times...")
            app_timeline = enriched_logs.groupBy('cluster_id', 'app_number').agg(
                spark_min('timestamp').alias('start_time'),
                spark_max('timestamp').alias('end_time')
            ).withColumn(
                'application_id',
                concat_ws('_', col('cluster_id'), col('app_number'))
            )
            
            # Convert to pandas for easier manipulation
            print("Converting results to pandas...")
            timeline_df = app_timeline.toPandas()
            
            # Parse timestamps
            timeline_df['start_datetime'] = pd.to_datetime(timeline_df['start_time'], format='%y/%m/%d %H:%M:%S', errors='coerce')
            timeline_df['end_datetime'] = pd.to_datetime(timeline_df['end_time'], format='%y/%m/%d %H:%M:%S', errors='coerce')
            
            # Calculate duration in seconds
            timeline_df['duration_seconds'] = (timeline_df['end_datetime'] - timeline_df['start_datetime']).dt.total_seconds()
            
            # Create output directory
            output_dir = Path('data/output')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Output 1: Timeline data
            print(f"Writing {output_prefix}_timeline.csv...")
            timeline_output = timeline_df[['cluster_id', 'application_id', 'app_number', 'start_time', 'end_time']].copy()
            timeline_output.to_csv(output_dir / f'{output_prefix}_timeline.csv', index=False)
            
            # Output 2: Cluster summary
            print(f"Writing {output_prefix}_cluster_summary.csv...")
            cluster_summary = timeline_df.groupby('cluster_id').agg({
                'application_id': 'count',
                'start_datetime': 'min',
                'end_datetime': 'max'
            }).rename(columns={
                'application_id': 'num_applications',
                'start_datetime': 'cluster_first_app',
                'end_datetime': 'cluster_last_app'
            }).reset_index()
            cluster_summary['cluster_first_app'] = cluster_summary['cluster_first_app'].dt.strftime('%Y-%m-%d %H:%M:%S')
            cluster_summary['cluster_last_app'] = cluster_summary['cluster_last_app'].dt.strftime('%Y-%m-%d %H:%M:%S')
            cluster_summary.to_csv(output_dir / f'{output_prefix}_cluster_summary.csv', index=False)
            
            # Output 3: Summary statistics
            print(f"Writing {output_prefix}_stats.txt...")
            total_clusters = len(cluster_summary)
            total_apps = len(timeline_df)
            avg_apps_per_cluster = total_apps / total_clusters
            
            with open(output_dir / f'{output_prefix}_stats.txt', 'w') as f:
                f.write(f"Total unique clusters: {total_clusters}\n")
                f.write(f"Total applications: {total_apps}\n")
                f.write(f"Average applications per cluster: {avg_apps_per_cluster:.2f}\n\n")
                f.write("Most heavily used clusters:\n")
                for _, row in cluster_summary.nlargest(10, 'num_applications').iterrows():
                    f.write(f"  Cluster {row['cluster_id']}: {row['num_applications']} applications\n")
            
            print("\n" + "="*60)
            print("CLUSTER USAGE ANALYSIS RESULTS")
            print("="*60)
            print(f"Total unique clusters: {total_clusters}")
            print(f"Total applications: {total_apps}")
            print(f"Average applications per cluster: {avg_apps_per_cluster:.2f}")
            print("\nMost heavily used clusters:")
            for _, row in cluster_summary.nlargest(5, 'num_applications').iterrows():
                print(f"  Cluster {row['cluster_id']}: {row['num_applications']} applications")
            print("="*60)
            
            # Save timeline_df for visualization
            timeline_df.to_csv(output_dir / 'problem2_timeline_df.csv', index=False)
            
        finally:
            spark.stop()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Load data if we didn't just create it
    if args.skip_spark:
        cluster_summary = pd.read_csv(Path('data/output') / f'{output_prefix}_cluster_summary.csv')
        timeline_df = pd.read_csv(Path('data/output') / 'problem2_timeline_df.csv')
        timeline_df['cluster_id'] = timeline_df['cluster_id'].astype(str)
    
    # Visualization 1: Bar chart - Applications per cluster
    print(f"Creating {output_prefix}_bar_chart.png...")
    plt.figure(figsize=(12, 8))
    cluster_summary_sorted = cluster_summary.sort_values('num_applications', ascending=False)
    
    ax = sns.barplot(
        data=cluster_summary_sorted,
        x='cluster_id',
        y='num_applications',
        palette='viridis'
    )
    
    # Add value labels on top of bars
    for i, (idx, row) in enumerate(cluster_summary_sorted.iterrows()):
        ax.text(i, row['num_applications'], f'{int(row["num_applications"])}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.title('Number of Applications per Cluster', fontsize=16, fontweight='bold')
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('Number of Applications', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_dir = Path('data/output')
    plt.savefig(output_dir / f'{output_prefix}_bar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Visualization 2: Density plot for largest cluster
    print(f"Creating {output_prefix}_density_plot.png...")
    
    # Find the cluster with most applications
    largest_cluster_id = cluster_summary_sorted.iloc[0]['cluster_id']
    largest_cluster_data = timeline_df[timeline_df['cluster_id'] == str(largest_cluster_id)]
    
    # Filter out invalid durations
    valid_durations = largest_cluster_data[
        (largest_cluster_data['duration_seconds'] > 0) &
        (largest_cluster_data['duration_seconds'] < 86400)  # Less than 1 day
    ]['duration_seconds']
    
    plt.figure(figsize=(12, 8))
    
    # Create histogram with KDE
    ax = sns.histplot(
        valid_durations,
        bins=50,
        kde=True,
        stat='density',
        edgecolor='black',
        linewidth=0.5
    )
    
    # Set log scale on x-axis
    ax.set_xscale('log')
    ax.set_xlabel('Job Duration (seconds)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    
    # Set custom x-axis ticks for log scale
    import numpy as np
    ticks = np.logspace(np.log10(valid_durations.min()), np.log10(valid_durations.max()), 10)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{int(t)}' for t in ticks])
    
    plt.title(f'Job Duration Distribution - Cluster {largest_cluster_id}\n(n={len(valid_durations)} applications)', 
              fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{output_prefix}_density_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization complete!")
    print(f"\nOutput files written to data/output/")
    print(f"  - {output_prefix}_timeline.csv")
    print(f"  - {output_prefix}_cluster_summary.csv")
    print(f"  - {output_prefix}_stats.txt")
    print(f"  - {output_prefix}_bar_chart.png")
    print(f"  - {output_prefix}_density_plot.png")


if __name__ == '__main__':
    main()

