from pyspark import pipelines as dp
from pyspark.sql.types import (
    StructType, StructField, StringType,
    IntegerType, DoubleType, TimestampType
)

# Lokasi file CSV yang sudah di-upload ke Volume
SOURCE_PATH = "/Volumes/workspace/lakeschema/online_retail/"

# Schema data Online Retail
retail_schema = StructType([
    StructField("InvoiceNo", StringType(), True),
    StructField("StockCode", StringType(), True),
    StructField("Description", StringType(), True),
    StructField("Quantity", IntegerType(), True),
    StructField("InvoiceDate", TimestampType(), True),
    StructField("UnitPrice", DoubleType(), True),
    StructField("CustomerID", StringType(), True),
    StructField("Country", StringType(), True)
])

# ========================================
# BRONZE LAYER - Incremental Ingestion
# ========================================

@dp.table(
    name="bronze_online_retail",
    comment="Data mentah transaksi Online Retail dari CSV secara incremental"
)
def bronze_online_retail():
    return (
        spark.readStream
        .format("cloudFiles")
        .schema(retail_schema)
        .option("cloudFiles.format", "csv")
        .option("header", "true")
        .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
        .load(SOURCE_PATH)
    )

# ========================================
# SILVER LAYER - DATA CLEANING
# ========================================

from pyspark.sql import functions as F

@dp.materialized_view(
    name="silver_online_retail",
    comment="Silver layer - cleaned and validated retail transactions"
)
def silver_online_retail():

    df = spark.read.table("bronze_online_retail")

    df = (
        df
        .withColumn("CustomerID", F.col("CustomerID").cast("string"))
        .withColumn("Description", F.trim(F.col("Description")))
        .withColumn(
            "TransactionType",
            F.when(
                F.upper(F.col("InvoiceNo")).startswith("C"),
                "Cancellation"
            )
            .when(F.col("Quantity") < 0, "Return")
            .otherwise("Sale")
        )
        .withColumn(
            "TotalAmount",
            F.col("Quantity") * F.col("UnitPrice")
        )
        .filter(F.col("InvoiceNo").isNotNull())
        .filter(F.col("StockCode").isNotNull())
        .filter(F.col("Quantity").isNotNull())
        .filter(F.col("UnitPrice").isNotNull())
        .filter(F.col("UnitPrice") >= 0)
    )

    # Validasi deskripsi produk
    df = df.withColumn(
        "IsValidProduct",
        F.col("Description").isNotNull()
        & (F.length(F.col("Description")) > 0)
        & F.col("Description").rlike("[A-Za-z]")
        & ~F.lower(F.col("Description")).rlike(
            r"^(\?+|missing\b|lost\b|damages?\b)"
        )
    )

    # Validasi transaksi penjualan
    df = df.withColumn(
        "IsValidSale",
        (F.col("TransactionType") == "Sale")
        & (F.col("Quantity") > 0)
        & (F.col("UnitPrice") > 0)
        & F.col("IsValidProduct")
    )

    return df

# ========================================
# GOLD LAYER - DAILY SALES AGGREGATION
# ========================================

@dp.materialized_view(
    name="gold_online_retail",
    comment="Gold layer - ringkasan penjualan harian Online Retail"
)
def gold_online_retail():

    df = spark.read.table("silver_online_retail")

    return (
        df
        .filter(F.col("IsValidSale") == True)
        .withColumn(
            "TransactionDate",
            F.to_date(F.col("InvoiceDate"))
        )
        .groupBy("TransactionDate", "Country")
        .agg(
            F.countDistinct("InvoiceNo").alias("TotalTransactions"),
            F.sum("Quantity").alias("TotalItemsSold"),
            F.round(
                F.sum("TotalAmount"), 2
            ).alias("TotalRevenue")
        )
    )

@dp.materialized_view(
    name="gold_online_retail_products",
    comment="Gold layer - valid product sales aggregation"
)
def gold_online_retail_products():

    df = spark.read.table("silver_online_retail")

    return (
        df
        .filter(F.col("IsValidSale") == True)
        .groupBy("StockCode", "Description")
        .agg(
            F.sum("Quantity").alias("TotalItemsSold"),
            F.countDistinct("InvoiceNo").alias("TotalTransactions"),
            F.round(
                F.sum("TotalAmount"), 2
            ).alias("TotalRevenue")
        )
    )

