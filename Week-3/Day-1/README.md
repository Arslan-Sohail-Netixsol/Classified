# Superstore Dataset Setup Guide

This guide provides the steps to set up the Superstore dataset in PostgreSQL. 
It bypasses pgAdmin's visual Import/Export bugs (such as Escape character and Encoding issues) by using direct SQL queries.

## 1. Create the Table
Open a Query Tool in your database and run the following SQL to create the `superstore` table. We use `VARCHAR` for all fields initially to prevent strict date formatting or numeric errors during the import process.

```sql
DROP TABLE IF EXISTS superstore;

CREATE TABLE superstore (
    row_id VARCHAR(50),
    order_id VARCHAR(50),
    order_date VARCHAR(50),
    ship_date VARCHAR(50),
    ship_mode VARCHAR(50),
    customer_id VARCHAR(50),
    customer_name VARCHAR(100),
    segment VARCHAR(50),
    country VARCHAR(100),
    city VARCHAR(100),
    state VARCHAR(100),
    postal_code VARCHAR(50),
    region VARCHAR(50),
    product_id VARCHAR(50),
    category VARCHAR(50),
    sub_category VARCHAR(50),
    product_name VARCHAR(255),
    sales VARCHAR(50),
    quantity VARCHAR(50),
    discount VARCHAR(50),
    profit VARCHAR(50)
);
```

## 2. Import the CSV Data
Use the PostgreSQL `COPY` command to import the CSV file quickly and safely. 

*Note: Ensure the file path below exactly matches the location of your CSV file on your computer.*

```sql
COPY superstore 
FROM 'D:\data sets\day 1 week 3\Sample - Superstore.csv' 
WITH (FORMAT csv, HEADER true, ENCODING 'WIN1252');
```

## 3. Verify the Import
Run these queries one by one to ensure your data was imported successfully:

**Count Total Rows (Should be 9994)**
```sql
SELECT COUNT(*) FROM superstore;
```

**Preview First 10 Rows**
```sql
SELECT * FROM superstore LIMIT 10;
```

**Check Table Structure (Columns & Types)**
```sql
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'superstore';
```
