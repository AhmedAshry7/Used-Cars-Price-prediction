import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pyspark
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, LongType

pyspark_home = os.path.dirname(pyspark.__file__)

os.environ['SPARK_HOME'] = pyspark_home
os.environ['PATH'] += os.pathsep + os.path.join(pyspark_home, 'bin')
# Manually point to your Java and Hadoop locations
os.environ['JAVA_HOME'] = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.18.8-hotspot"
os.environ['HADOOP_HOME'] = r"C:\hadoop"

# Force the 'bin' folders into the session path
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['JAVA_HOME'], 'bin')
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['HADOOP_HOME'], 'bin')

# Tell Spark exactly which python to use
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

if "spark" in globals():
    spark.stop()
spark = SparkSession.builder.appName("CarValidationScript").config("spark.sql.ansi.enabled", "false").getOrCreate()
sc = spark.sparkContext



class DataValidator:

    def validate_huge_dataset(df):
        
        total_rows = df.count()
        print(f"Total Rows: {total_rows}")

        # 2. Define the columns we EXPECT to be numeric
        numeric_cols = ["price", "year", "odometer", "lat", "long"]
        
        # 3. Optimized Single-Pass Validation
        # We calculate Nulls, NaNs, and Type-Casting Failures in one go
        validation_logic = []
        
        for c in df.columns:
            # 1. Define what "Missing" looks like in this dataset
            # We check for actual Nulls, the string "NULL", or empty strings
            is_null = F.col(c).isNull() | (F.col(c) == "NULL") | (F.col(c) == "")
            validation_logic.append(F.count(F.when(is_null, c)).alias(f"{c}_missing"))
            
            # 2. Check for Type Errors (if column is supposed to be numeric)
            if c in numeric_cols:
                # We attempt to cast. If it results in NULL but wasn't originally NULL, it's a type error.
                is_invalid_type = F.col(c).isNotNull() & (F.col(c) != "") & F.col(c).cast("double").isNull()
                validation_logic.append(F.count(F.when(is_invalid_type, c)).alias(f"{c}_type_error"))

        print("Analyzing data quality... This scan handles the 'huge' data efficiently.")
        # Single action: one scan through the whole file
        results = df.select(validation_logic).collect()[0].asDict()

        # 3. Print the Report
        print("\n" + "="*60)
        print(f"{'Column Name':<15} | {'Missing %':<10} | {'Type Errors'}")
        print("-" * 60)
        
        for c in df.columns:
            missing = results.get(f"{c}_missing", 0)
            missing_pct = (missing / total_rows) * 100
            
            # Pull type errors for numeric columns
            type_errors = results.get(f"{c}_type_error", 0)
            type_err_str = f"{type_errors} rows" if c in numeric_cols else "N/A"
            
            print(f"{c:<15} | {missing_pct:8.2f}% | {type_err_str}")


    def check_schema_and_types(df, expected_schema):
        print("\n--- Metadata & Type Conformance Check ---")
        
        # Quick Name Check
        if df.columns == expected_schema.names:
            print("✅ Column Names & Order: Match.")
        else:
            print(f"⚠️ Mismatch! Expected {len(expected_schema.names)} cols, found {len(df.columns)}.")

        # Optimized Type Check using expr()
        validation_exprs = []
        numeric_fields = [f for f in expected_schema.fields if not isinstance(f.dataType, StringType)]
        
        for field in numeric_fields:
            col_name = field.name
            data_type = field.dataType.simpleString() # e.g., 'bigint' or 'double'
            
            # We use F.expr to call the SQL version of try_cast
            # If the cast fails (like for '3009548743' to INT), it returns NULL
            # but does NOT crash the job.
            invalid_mask = F.col(col_name).isNotNull() & \
                        (F.col(col_name) != "") & \
                        F.expr(f"try_cast({col_name} as {data_type})").isNull()
            
            validation_exprs.append(F.count(F.when(invalid_mask, True)).alias(col_name))

        print("Scanning data for type mismatches (using Long/BigInt for large values)...")
        results = df.select(validation_exprs).collect()[0].asDict()
        
        for col, error_count in results.items():
            if error_count == 0:
                print(f"✅ {col:<15}: All rows conform to schema.")
            else:
                print(f"❌ {col:<15}: {error_count} rows fail casting (likely overflow or text).")

    def get_unique_values(df, columns_to_check):
        print("\n--- Unique Values Report to check for inconsistencies ---")
        for col_name in columns_to_check:
            print(f"\nUnique values for {col_name}:")
            
            # 1. Get distinct values
            # 2. Drop nulls so they don't clutter the list
            # 3. Sort them so it's readable
            unique_df = df.select(col_name).distinct().na.drop().orderBy(col_name)
            
            # If the list is small, show it all. If huge, show top 20.
            unique_count = unique_df.count()
            print(f"Total Unique: {unique_count}")
            unique_df.show(truncate=False)

    def check_price_accuracy(df):
        total_rows = df.count()
        # --- 1. Check Price <= 0 ---
        # We filter for values that are logically 0 or less
        invalid_price_count = df.filter(F.col("price") <= 0).count()
        print(f"Rows with price <= 0: {invalid_price_count} ({ (invalid_price_count/total_rows)*100 :.2f}%)")

    def audit_column_quality(df, columns_list):
        """
        Scans a list of columns for missing values and 'other' entries 
        using a single-pass aggregation for big data efficiency.
        """
        total_rows = df.count()
        if total_rows == 0:
            print("Dataset is empty.")
            return

        # 1. Build the aggregation expressions
        # We look for standard NULLs, empty strings, and the specific string 'other'
        agg_exprs = []
        for col_name in columns_list:
            # Standard Nulls / Empty
            is_null_cond = F.col(col_name).isNull() | (F.col(col_name) == "") | (F.col(col_name) == "NULL")
            agg_exprs.append(F.count(F.when(is_null_cond, 1)).alias(f"{col_name}_nulls"))
            
            # 'Other' entries (case-insensitive check)
            is_other_cond = F.lower(F.col(col_name)) == "other"
            agg_exprs.append(F.count(F.when(is_other_cond, 1)).alias(f"{col_name}_others"))

        # 2. Execute the scan once
        print(f"Auditing {len(columns_list)} columns across {total_rows} rows...")
        results = df.select(agg_exprs).collect()[0].asDict()

        # 3. Print the report
        print("\n" + "="*75)
        print(f"{'Column':<20} | {'Null %':<12} | {'Other %':<12} | {'Total Dirty %'}")
        print("-" * 75)

        for col_name in columns_list:
            null_count = results[f"{col_name}_nulls"]
            other_count = results[f"{col_name}_others"]
            
            null_pct = (null_count / total_rows) * 100
            other_pct = (other_count / total_rows) * 100
            total_dirty_pct = ((null_count + other_count) / total_rows) * 100

            print(f"{col_name:<20} | {null_pct:>6.2f}%      | {other_pct:>6.2f}%      | {total_dirty_pct:>6.2f}%")
        print("="*75)

    def clean_cylinders(df):
        total_rows = df.count()
        # --- Clean Cylinders ---
        # regexp_extract(column, regex, group) 
        # '\d+' looks for one or more digits. If it doesn't find any (like in 'other'), it returns an empty string.
        print("\nCleaning 'cylinders' column...")
        
        df_cleaned = df.withColumn(
            "cylinders_numeric", 
            F.regexp_extract(F.col("cylinders"), r"(\d+)", 1).cast("int")
        )

        # --- 3. Calculate New Null Percentage for Cylinders ---
        # Since "other" doesn't have digits, cast("int") turns it into NULL automatically.
        stats = df_cleaned.select(
            F.count(F.when(F.col("cylinders_numeric").isNull(), 1)).alias("null_count")
        ).collect()[0]

        new_null_count = stats["null_count"]
        new_null_pct = (new_null_count / total_rows) * 100

        print(f"⚙️ Cylinders (Numeric) Missing Percentage: {new_null_pct:.2f}%")
        print(f"   (Includes original nulls and rows that were 'other')")

        # Show the transformation result
        print("\nSample of transformed cylinders:")
        df_cleaned.select("cylinders", "cylinders_numeric").distinct().show()
        
        return df_cleaned

    def analyze_outliers_and_plot(df, numeric_cols):
        total_rows = df.count()
        if total_rows == 0:
            print("Dataset is empty.")
            return

        for col_name in numeric_cols:
            print("\n" + "="*50)
            print(f"Analyzing Column: {col_name}")
            print("="*50)
            
            # --- 1. Calculate IQR using approxQuantile ---
            # Parameters: (column_name, [probabilities], relative_error)
            # 0.01 error means it's 99% accurate but much faster than an exact sort
            quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.01)
            
            if len(quantiles) < 2:
                print(f"Not enough data in {col_name} to calculate quartiles.")
                continue
                
            q1, q3 = quantiles[0], quantiles[1]
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Count the outliers
            outliers_count = df.filter(
                (F.col(col_name) < lower_bound) | (F.col(col_name) > upper_bound)
            ).count()
            
            outliers_pct = (outliers_count / total_rows) * 100
            
            print(f"Q1: {q1:.2f} | Q3: {q3:.2f} | IQR: {iqr:.2f}")
            print(f"Valid Range: [{lower_bound:.2f} to {upper_bound:.2f}]")
            print(f"Outliers Found: {outliers_count} rows ({outliers_pct:.2f}%)")

            # --- 2. Build the Big Data Histogram ---
            print(f"Generating histogram for {col_name}...")
            
            # We must drop NULLs and cast to Double for the RDD histogram function to work
            clean_df = df.filter(F.col(col_name).isNotNull()).select(F.col(col_name).cast("double"))
            
            try:
                # Spark calculates the bins and counts distributedly! 
                # 20 represents the number of bins we want.
                rdd_data = clean_df.rdd.flatMap(lambda x: x)
                bins, counts = rdd_data.histogram(20)
                
                # Plotting the results using Matplotlib
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # Calculate the width of each bin for plotting
                bin_widths = [bins[i+1] - bins[i] for i in range(len(bins)-1)]
                
                # Draw the bar chart using the pre-aggregated Spark data
                ax.bar(bins[:-1], counts, width=bin_widths, align='edge', edgecolor='black', alpha=0.75)
                
                ax.set_title(f"Distribution of {col_name}")
                ax.set_xlabel(col_name)
                ax.set_ylabel("Frequency")
                
                # Use tight_layout to ensure labels aren't cut off
                plt.tight_layout()
                
                # Save the plot (you can also use plt.show() if running in a notebook)
                file_name = f"Figures/histogram_{col_name}.png"
                plt.savefig(file_name)
                print(f"✅ Histogram saved as {file_name}")
                
                # Clear the plot from memory so they don't overlap
                plt.close(fig)
                
            except Exception as e:
                print(f"❌ Failed to generate histogram for {col_name}: {e}")




# 1. Read everything as String to prevent casting crashes
# We use 'multiLine' because 'desc' columns often have newlines
# We use 'escape' to handle quotes within descriptions
df = spark.read.csv("data/craigslistVehicles.csv", header=True, inferSchema=False, multiLine=True, escape='"')
# Print the top 10 rows
df.show(5)

validator=DataValidator()
validator.validate_huge_dataset(df)


actual_dtypes = dict(df.dtypes)
print("Actual Data Types:")
for col, dtype in actual_dtypes.items():
    print(f"{col}: {dtype}")

# 1. Define your Expected Schema

schema = StructType([
    StructField("url", StringType(), True),
    StructField("city", StringType(), True),
    StructField("city_url", StringType(), True),
    StructField("price", LongType(), True),
    StructField("year", IntegerType(), True),
    StructField("manufacturer", StringType(), True),
    StructField("make", StringType(), True),
    StructField("condition", StringType(), True),
    StructField("cylinders", StringType(), True),
    StructField("fuel", StringType(), True),
    StructField("odometer", DoubleType(), True),
    StructField("title_status", StringType(), True),
    StructField("transmission", StringType(), True),
    StructField("VIN", StringType(), True),
    StructField("drive", StringType(), True),
    StructField("size", StringType(), True),
    StructField("type", StringType(), True),
    StructField("paint_color", StringType(), True),
    StructField("image_url", StringType(), True),
    StructField("desc", StringType(), True),
    StructField("lat", DoubleType(), True),
    StructField("long", DoubleType(), True)
])

validator.check_schema_and_types(df, schema)

df_casted = df
for field in schema.fields:
    col_name = field.name
    target_type = field.dataType
    # Apply the cast from the schema blueprint to the column
    df_casted = df_casted.withColumn(col_name, F.col(col_name).cast(target_type))

# Now df_casted has the actual Longs, Integers, and Doubles.
# Let's verify:
df_casted.printSchema()
validator.get_unique_values(df_casted, ["manufacturer", "year", "condition", "fuel", "cylinders", "title_status", "transmission", "drive", "type"])  
validator.check_price_accuracy(df_casted)      

columns_to_audit = ["manufacturer", "year", "fuel", "condition", "cylinders", "title_status", "transmission", "drive", "type", "city" ]
validator.audit_column_quality(df_casted, columns_to_audit)
df_casted = validator.clean_cylinders(df_casted)

numeric_columns_to_check = ["price", "odometer", "lat", "long"]
validator.analyze_outliers_and_plot(df_casted, numeric_columns_to_check)