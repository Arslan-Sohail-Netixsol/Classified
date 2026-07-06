-- 1. Create the Table


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

-- 2. Import the CSV Data
-- Make sure the path matches your local file location exactly
COPY superstore 
FROM 'D:\data sets\day 1 week 3\Sample - Superstore.csv' 
WITH (FORMAT csv, HEADER true, ENCODING 'WIN1252');

-- 3. Verify the Import
-- Count Total Rows (Should be 9994)
SELECT COUNT(*) FROM superstore;

-- Preview First 10 Rows
SELECT * FROM superstore LIMIT 10;

-- Check Table Structure (Columns & Types)
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'superstore';
