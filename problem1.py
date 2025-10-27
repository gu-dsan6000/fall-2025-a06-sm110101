"""
Problem 1: Log Level Distribution Analysis

This script analyzes the distribution of log levels (INFO, WARN, ERROR, DEBUG)
across all Spark cluster log files and generates three output files:


**Expected output 1 (counts):**
```
log_level,count
INFO,125430
WARN,342
ERROR,89
DEBUG,12
```

**Expected output 2 (sample):**
```
log_entry,log_level
"17/03/29 10:04:41 INFO ApplicationMaster: Registered signal handlers",INFO
"17/03/29 10:04:42 WARN YarnAllocator: Container request...",WARN
...
```

**Expected output 3 (summary):**
```
Total log lines processed: 3,234,567
Total lines with log levels: 3,100,234
Unique log levels found: 4

Log level distribution:
  INFO  :    125,430 (40.45%)
  WARN  :        342 ( 0.01%)
  ...
```
"""

import argparse
import sys
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    regexp_extract,
    rand,
    count,
    when,
    lit,
)
from pyspark.sql.types import StringType


def extract_log_level(line):
    """Extract log level from a log line."""
    import re
    pattern = r'^\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2} (INFO|WARN|ERROR|DEBUG)'
    match = re.match(pattern, line)
    return match.group(1) if match else None


def main():
    parser = argparse.ArgumentParser(description='Analyze log level distribution')
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
    
    args = parser.parse_args()
    
    if args.local:
        data_path = "data/sample/"
        output_prefix = "problem1"
        print("Running in LOCAL mode using sample data...")
    elif args.master_url:
        if not args.net_id:
            print("Error: --net-id is required when running on cluster")
            sys.exit(1)
        data_path = f"s3a://{args.net_id}-assignment-spark-cluster-logs/data/"
        output_prefix = "problem1"
        print(f"Running on cluster using S3 data: {data_path}")
    else:
        print("Error: Either provide master_url or use --local flag")
        sys.exit(1)
    
    # Initialize Spark session
    builder = (
        SparkSession.builder
        .appName("Log Level Distribution Analysis")
    )
    
    if args.master_url and not args.local:
        builder = (
            builder
            .master(args.master_url)
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.InstanceProfileCredentialsProvider")
            # Performance settings
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            # Memory settings
            .config("spark.executor.memory", "4g")
            .config("spark.driver.memory", "2g")
        )
    
    spark = builder.getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    try:
        # Read all log files
        print(f"Reading log files from: {data_path}")
        logs_df = spark.read.text(f"{data_path}*/container_*.log")
        
        # Extract log level using regex
        logs_with_levels = logs_df.withColumn(
            'log_level',
            regexp_extract(
                'value',
                r'^\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2} (INFO|WARN|ERROR|DEBUG)',
                1
            )
        ).withColumn(
            'has_level',
            col('log_level') != ''
        )
        
        # Filter to only lines with valid log levels
        valid_logs = logs_with_levels.filter(col('has_level') == True)
        
        # Count occurrences of each log level
        print("Counting log levels...")
        level_counts = valid_logs.groupBy('log_level').count().orderBy('count', ascending=False)
        level_counts.cache()
        
        # Get total counts
        counts_dict = {row['log_level']: row['count'] for row in level_counts.collect()}
        total_with_levels = valid_logs.count()
        total_lines = logs_df.count()
        
        # Output 1: problem1_counts.csv
        print(f"Writing {output_prefix}_counts.csv...")
        counts_df = level_counts.toPandas()
        counts_df.columns = ['log_level', 'count']
        
        output_dir = Path('data/output')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        counts_df.to_csv(output_dir / f'{output_prefix}_counts.csv', index=False)
        
        # Output 2: problem1_sample.csv - 10 random samples
        print(f"Writing {output_prefix}_sample.csv...")
        sample_df = valid_logs.select('value', 'log_level').orderBy(rand()).limit(10)
        sample_pandas = sample_df.toPandas()
        sample_pandas.columns = ['log_entry', 'log_level']
        sample_pandas.to_csv(output_dir / f'{output_prefix}_sample.csv', index=False)
        
        # Output 3: problem1_summary.txt
        print(f"Writing {output_prefix}_summary.txt...")
        with open(output_dir / f'{output_prefix}_summary.txt', 'w') as f:
            f.write(f"Total log lines processed: {total_lines:,}\n")
            f.write(f"Total lines with log levels: {total_with_levels:,}\n")
            f.write(f"Unique log levels found: {len(counts_dict)}\n")
            f.write("\n")
            f.write("Log level distribution:\n")
            
            for level in ['INFO', 'WARN', 'ERROR', 'DEBUG']:
                count = counts_dict.get(level, 0)
                percentage = (count / total_with_levels * 100) if total_with_levels > 0 else 0
                f.write(f"  {level:6s}: {count:8,} ({percentage:6.2f}%)\n")
        
        # Print summary to console
        print("\n" + "="*60)
        print("LOG LEVEL DISTRIBUTION ANALYSIS RESULTS")
        print("="*60)
        print(f"Total log lines processed: {total_lines:,}")
        print(f"Total lines with log levels: {total_with_levels:,}")
        print(f"Unique log levels found: {len(counts_dict)}")
        print("\nLog level distribution:")
        for level in ['INFO', 'WARN', 'ERROR', 'DEBUG']:
            count = counts_dict.get(level, 0)
            percentage = (count / total_with_levels * 100) if total_with_levels > 0 else 0
            print(f"  {level:6s}: {count:8,} ({percentage:6.2f}%)")
        print("="*60)
        print(f"\nOutput files written to data/output/")
        print(f"  - {output_prefix}_counts.csv")
        print(f"  - {output_prefix}_sample.csv")
        print(f"  - {output_prefix}_summary.txt")
        
    finally:
        spark.stop()


if __name__ == '__main__':
    main()

