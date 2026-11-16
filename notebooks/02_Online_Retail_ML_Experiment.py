# Databricks notebook source
# DBTITLE 1,Membaca data dari Silver
from pyspark.sql import functions as F

df_silver = spark.table(
    "workspace.lakeschema.silver_online_retail"
)

print("Total data Silver:", df_silver.count())
display(df_silver.limit(10))


# COMMAND ----------

# DBTITLE 1,Menyiapkan data training
# Menyiapkan data klasifikasi harga produk
df_ml = (
    df_silver
    .filter(F.col("IsValidSale") == True)
    .filter(F.col("IsValidProduct") == True)
    .filter(
        ~F.upper(F.trim(F.col("StockCode"))).isin(
            "DOT", "POST", "M"
        )
    )
    .select(
        "StockCode",
        "Description",
        "UnitPrice",
        "InvoiceDate"
    )
)

print("Jumlah baris calon training:", df_ml.count())

display(df_ml.limit(10))


# COMMAND ----------

# DBTITLE 1,Analisis distribusi harga
display(
    df_ml.select("UnitPrice").summary(
        "count", "min", "25%", "50%", "75%",
        "max", "mean", "stddev"
    )
)

# Periksa jumlah produk unik
print(
    "Produk unik berdasarkan StockCode:",
    df_ml.select("StockCode").distinct().count()
)

print(
    "Deskripsi produk unik:",
    df_ml.select("Description").distinct().count()
)


# COMMAND ----------

# DBTITLE 1,Menyiapkan data per produk

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Pastikan StockCode valid
df_ml_valid = (
    df_ml
    .filter(F.col("StockCode").isNotNull())
    .filter(F.length(F.trim(F.col("StockCode"))) > 0)
)

# Hitung frekuensi deskripsi per produk
description_counts = (
    df_ml_valid
    .groupBy("StockCode", "Description")
    .agg(
        F.count("*").alias("DescriptionFrequency")
    )
)

# Pilih deskripsi yang paling sering muncul
window_spec = Window.partitionBy("StockCode").orderBy(
    F.desc("DescriptionFrequency"),
    F.asc("Description")
)

product_descriptions = (
    description_counts
    .withColumn(
        "rn",
        F.row_number().over(window_spec)
    )
    .filter(F.col("rn") == 1)
    .select("StockCode", "Description")
)

# Hitung harga median dan jumlah transaksi
product_prices = (
    df_ml_valid
    .groupBy("StockCode")
    .agg(
        F.percentile_approx(
            "UnitPrice", 0.5
        ).alias("ProductPrice"),
        F.count("*").alias("TransactionCount")
    )
)

# Gabungkan hasil agregasi
df_products = (
    product_prices
    .join(
        product_descriptions,
        on="StockCode",
        how="inner"
    )
    .select(
        "StockCode",
        "Description",
        "ProductPrice",
        "TransactionCount"
    )
)

print("Total produk:", df_products.count())
display(df_products.limit(10))


# COMMAND ----------

# DBTITLE 1,Membagi data training dan testing
# Split berdasarkan deskripsi produk
# Agar deskripsi identik tidak masuk train dan test
df_products_clean = (
    df_products
    .withColumn(
        "DescriptionClean",
        F.lower(
            F.trim(
                F.regexp_replace(
                    F.col("Description"),
                    r"\s+",
                    " "
                )
            )
        )
    )
    .filter(F.col("DescriptionClean").isNotNull())
    .filter(F.length(F.col("DescriptionClean")) > 0)
)

df_split = df_products_clean.withColumn(
    "split_group",
    F.pmod(F.xxhash64("DescriptionClean"), F.lit(10))
)

df_train = df_split.filter(F.col("split_group") < 8)
df_test = df_split.filter(F.col("split_group") >= 8)

print("Jumlah training:", df_train.count())
print("Jumlah testing:", df_test.count())

# COMMAND ----------

# DBTITLE 1,Membuat label klasifikasi harga
# Threshold dihitung dari training
threshold = df_train.approxQuantile(
    "ProductPrice", [0.5], 0.001
)[0]

train_labeled = df_train.withColumn(
    "label",
    F.when(F.col("ProductPrice") > threshold, 1).otherwise(0)
)

test_labeled = df_test.withColumn(
    "label",
    F.when(F.col("ProductPrice") > threshold, 1).otherwise(0)
)

print("Batas kategori harga:", threshold)

print("Distribusi training:")
display(train_labeled.groupBy("label").count())

print("Distribusi testing:")
display(test_labeled.groupBy("label").count())

# COMMAND ----------

# DBTITLE 1,Memastikan tidak ada deskripsi identik

train_desc = train_labeled.select(
    "DescriptionClean"
).distinct()

test_desc = test_labeled.select(
    "DescriptionClean"
).distinct()

overlap = train_desc.intersect(test_desc).count()

print("Jumlah deskripsi overlap:", overlap)
print("Jumlah training:", train_labeled.count())
print("Jumlah testing:", test_labeled.count())


# COMMAND ----------

# DBTITLE 1,Training model TF-IDF
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Mengubah data Spark ke Pandas
train_pd = train_labeled.select(
    "DescriptionClean", "label"
).toPandas()

test_pd = test_labeled.select(
    "DescriptionClean", "label"
).toPandas()

X_train = train_pd["DescriptionClean"].fillna("")
y_train = train_pd["label"].astype(int)

X_test = test_pd["DescriptionClean"].fillna("")
y_test = test_pd["label"].astype(int)

# Pipeline Machine Learning
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=5000
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# Melatih model
model.fit(X_train, y_train)

print("Training model berhasil!")
print("Jumlah data training:", len(X_train))
print("Jumlah data testing:", len(X_test))


# COMMAND ----------

# DBTITLE 1,Evaluasi model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

y_pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall   :", recall_score(y_test, y_pred, zero_division=0))
print("F1-Score :", f1_score(y_test, y_pred, zero_division=0))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Harga Rendah", "Harga Tinggi"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# COMMAND ----------

# DBTITLE 1,Membandingkan dengan baseline

from sklearn.dummy import DummyClassifier

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_test)

print("Baseline Accuracy:",
      accuracy_score(y_test, baseline_pred))

print("Model Accuracy:",
      accuracy_score(y_test, y_pred))

print("Baseline F1:",
      f1_score(y_test, baseline_pred, zero_division=0))

print("Model F1:",
      f1_score(y_test, y_pred, zero_division=0))


# COMMAND ----------

# DBTITLE 1,Mencatat eksperimen MLflow
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Menggunakan Unity Catalog sebagai registry
mlflow.set_registry_uri("databricks-uc")

# Mengubah pandas Series menjadi NumPy array
input_example = X_train.iloc[:3].to_numpy()

# Membuat signature sesuai input model
signature = infer_signature(
    input_example,
    model.predict(input_example)
)

with mlflow.start_run(
    run_name="TFIDF_LogisticRegression_OnlineRetail"
) as run:

    # Parameter
    mlflow.log_params({
        "algorithm": "LogisticRegression",
        "feature_extraction": "TF-IDF",
        "max_features": 5000,
        "ngram_range": "1-2",
        "price_threshold": float(threshold),
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    })

    # Evaluasi model
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "baseline_accuracy": accuracy_score(
            y_test, baseline_pred
        )
    })

    # Menyimpan model yang sudah dilatih
    mlflow.sklearn.log_model(
        sk_model=model,
        name="online_retail_classifier",
        input_example=input_example,
        signature=signature
    )

    print("MLflow Run ID:", run.info.run_id)

print("Model berhasil disimpan")
