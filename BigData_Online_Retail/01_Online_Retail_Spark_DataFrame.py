# Databricks notebook source
# DBTITLE 1,Baca Data online_retail
# 1. Membaca dataset Online Retail dari Volume
file_path = "/Volumes/workspace/lakeschema/online_retail/online_retail_2010_12.csv"

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
    .csv(file_path)
)

# 2. Memeriksa ukuran dataset
print("Jumlah baris:", df.count())
print("Jumlah kolom:", len(df.columns))
display(df.limit(10))

# 3. Memeriksa struktur dan tipe data
df.printSchema()

# 4. Memeriksa missing values
from pyspark.sql.functions import col, count, when

null_counts = df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in df.columns
])

display(null_counts)


# COMMAND ----------

# DBTITLE 1,Melakukan exploratory data dengan membuat tabel SQL menggunakan Spark.
# Membuat temporary view untuk analisis SQL
df.createOrReplaceTempView("online_retail_temp")

print("Temporary view online_retail_temp berhasil dibuat")


# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC     COUNT(*) AS total_baris,
# MAGIC     COUNT(DISTINCT InvoiceNo) AS total_invoice,
# MAGIC     COUNT(DISTINCT StockCode) AS total_produk,
# MAGIC     COUNT(DISTINCT Country) AS total_negara
# MAGIC FROM online_retail_temp;

# COMMAND ----------

# DBTITLE 1,Membuat tabel SQL permanen
# MAGIC %sql
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS workspace.lakeschema.online_retail_eda
# MAGIC USING DELTA
# MAGIC AS
# MAGIC SELECT *
# MAGIC FROM online_retail_temp;
# MAGIC

# COMMAND ----------

# DBTITLE 1,tahap 1 - Produk paling banyak terjual
# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC     StockCode,
# MAGIC     Description,
# MAGIC     SUM(Quantity) AS total_terjual
# MAGIC FROM workspace.lakeschema.online_retail_eda
# MAGIC WHERE Quantity > 0
# MAGIC   AND UnitPrice > 0
# MAGIC   AND InvoiceNo NOT LIKE 'C%'
# MAGIC GROUP BY StockCode, Description
# MAGIC ORDER BY total_terjual DESC
# MAGIC LIMIT 10;
# MAGIC

# COMMAND ----------

# DBTITLE 1,Tahap 2 - Negara dengan nilai penjualan tertinggi
# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC     Country,
# MAGIC     ROUND(SUM(Quantity * UnitPrice), 2) AS total_penjualan
# MAGIC FROM workspace.lakeschema.online_retail_eda
# MAGIC WHERE Quantity > 0
# MAGIC   AND UnitPrice > 0
# MAGIC   AND InvoiceNo NOT LIKE 'C%'
# MAGIC GROUP BY Country
# MAGIC ORDER BY total_penjualan DESC
# MAGIC LIMIT 10;
# MAGIC

# COMMAND ----------

# DBTITLE 1,Tahap 3 - Distribusi transaksi per hari
# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC     TO_DATE(InvoiceDate) AS tanggal,
# MAGIC     COUNT(DISTINCT InvoiceNo) AS jumlah_invoice,
# MAGIC     ROUND(SUM(Quantity * UnitPrice), 2) AS total_penjualan
# MAGIC FROM workspace.lakeschema.online_retail_eda
# MAGIC WHERE Quantity > 0
# MAGIC   AND UnitPrice > 0
# MAGIC   AND InvoiceNo NOT LIKE 'C%'
# MAGIC GROUP BY TO_DATE(InvoiceDate)
# MAGIC ORDER BY tanggal;
# MAGIC

# COMMAND ----------

# DBTITLE 1,Layer Bronze
# MAGIC %sql
# MAGIC
# MAGIC SELECT COUNT(*) AS total_baris
# MAGIC FROM workspace.lakeschema.bronze_online_retail;
# MAGIC

# COMMAND ----------

# DBTITLE 1,Layer Bronze-Silver
# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC     TransactionType,
# MAGIC     COUNT(*) AS jumlah_baris,
# MAGIC     ROUND(SUM(TotalAmount), 2) AS total_nilai
# MAGIC FROM workspace.lakeschema.silver_online_retail
# MAGIC GROUP BY TransactionType;
# MAGIC

# COMMAND ----------

# DBTITLE 1,Layer Bronze-Silver-Gold
# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC     TransactionDate,
# MAGIC     Country,
# MAGIC     TotalTransactions,
# MAGIC     TotalItemsSold,
# MAGIC     TotalRevenue
# MAGIC FROM workspace.lakeschema.gold_online_retail
# MAGIC ORDER BY TransactionDate
# MAGIC LIMIT 10;
# MAGIC

# COMMAND ----------

# DBTITLE 1,Kondisi saat semua data dimasukkan
# MAGIC %sql
# MAGIC
# MAGIC SELECT 'Bronze' AS layer, COUNT(*) AS total_baris
# MAGIC FROM workspace.lakeschema.bronze_online_retail
# MAGIC
# MAGIC UNION ALL
# MAGIC
# MAGIC SELECT 'Silver', COUNT(*)
# MAGIC FROM workspace.lakeschema.silver_online_retail
# MAGIC
# MAGIC UNION ALL
# MAGIC
# MAGIC SELECT 'Gold', COUNT(*)
# MAGIC FROM workspace.lakeschema.gold_online_retail;
# MAGIC