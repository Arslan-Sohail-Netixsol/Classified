CREATE SCHEMA IF NOT EXISTS analytics;

-- ==========================================
-- TASK 1 & 2: Analytics Layer & Chained SQL Pipeline
-- LAYER 1: Core Dimensions
-- ==========================================

-- dim_product
DROP VIEW IF EXISTS analytics.dim_product CASCADE;
CREATE VIEW analytics.dim_product AS
SELECT 
    p.productid,
    p.productnumber,
    p.name AS product_name,
    p.color,
    p.standardcost,
    p.listprice,
    p.safetystocklevel,
    p.reorderpoint,
    ps.name AS subcategory_name,
    pc.name AS category_name
FROM production.product p
LEFT JOIN production.productsubcategory ps ON p.productsubcategoryid = ps.productsubcategoryid
LEFT JOIN production.productcategory pc ON ps.productcategoryid = pc.productcategoryid;

-- dim_customer
DROP VIEW IF EXISTS analytics.dim_customer CASCADE;
CREATE VIEW analytics.dim_customer AS
SELECT 
    c.customerid,
    c.personid,
    c.storeid,
    c.territoryid,
    p.firstname || ' ' || COALESCE(p.middlename || ' ', '') || p.lastname AS customer_name,
    p.persontype,
    s.name AS store_name
FROM sales.customer c
LEFT JOIN person.person p ON c.personid = p.businessentityid
LEFT JOIN sales.store s ON c.storeid = s.businessentityid;

-- dim_territory
DROP VIEW IF EXISTS analytics.dim_territory CASCADE;
CREATE VIEW analytics.dim_territory AS
SELECT 
    t.territoryid,
    t.name AS territory_name,
    t.countryregioncode,
    t."group" AS territory_group,
    cr.name AS country_name
FROM sales.salesterritory t
LEFT JOIN person.countryregion cr ON t.countryregioncode = cr.countryregioncode;

-- dim_employee
DROP VIEW IF EXISTS analytics.dim_employee CASCADE;
CREATE VIEW analytics.dim_employee AS
SELECT 
    e.businessentityid AS employeeid,
    p.firstname || ' ' || COALESCE(p.middlename || ' ', '') || p.lastname AS employee_name,
    e.jobtitle,
    e.gender,
    e.hiredate,
    sp.salesquota,
    sp.bonus,
    sp.commissionpct
FROM humanresources.employee e
INNER JOIN person.person p ON e.businessentityid = p.businessentityid
LEFT JOIN sales.salesperson sp ON e.businessentityid = sp.businessentityid;

-- dim_vendor
DROP VIEW IF EXISTS analytics.dim_vendor CASCADE;
CREATE VIEW analytics.dim_vendor AS
SELECT 
    v.businessentityid AS vendorid,
    v.accountnumber,
    v.name AS vendor_name,
    v.creditrating,
    v.preferredvendorstatus,
    v.activeflag
FROM purchasing.vendor v;

-- dim_date
DROP VIEW IF EXISTS analytics.dim_date CASCADE;
CREATE VIEW analytics.dim_date AS
SELECT 
    d::date AS date,
    EXTRACT(YEAR FROM d) AS year,
    EXTRACT(QUARTER FROM d) AS quarter,
    EXTRACT(MONTH FROM d) AS month,
    TO_CHAR(d, 'Month') AS month_name,
    EXTRACT(DOW FROM d) AS day_of_week
FROM generate_series('2000-01-01'::timestamp, '2030-12-31'::timestamp, '1 day'::interval) d;


-- ==========================================
-- TASK 1 & 2: Analytics Layer & Chained SQL Pipeline
-- LAYER 2: Core Facts
-- ==========================================

-- fact_sales
DROP VIEW IF EXISTS analytics.fact_sales CASCADE;
CREATE VIEW analytics.fact_sales AS
SELECT 
    soh.salesorderid,
    sod.salesorderdetailid,
    soh.orderdate,
    soh.duedate,
    soh.shipdate,
    soh.status,
    soh.onlineorderflag,
    soh.customerid,
    soh.salespersonid,
    soh.territoryid,
    soh.billtoaddressid,
    sod.productid,
    sod.orderqty,
    sod.unitprice,
    sod.unitpricediscount,
    (sod.orderqty * sod.unitprice * (1 - sod.unitpricediscount)) AS linetotal,
    p.standardcost,
    ((sod.orderqty * sod.unitprice * (1 - sod.unitpricediscount)) - (sod.orderqty * p.standardcost)) AS profit
FROM sales.salesorderheader soh
INNER JOIN sales.salesorderdetail sod ON soh.salesorderid = sod.salesorderid
INNER JOIN production.product p ON sod.productid = p.productid;

-- fact_inventory
DROP VIEW IF EXISTS analytics.fact_inventory CASCADE;
CREATE VIEW analytics.fact_inventory AS
SELECT 
    i.productid,
    i.locationid,
    l.name AS location_name,
    i.shelf,
    i.bin,
    i.quantity
FROM production.productinventory i
INNER JOIN production.location l ON i.locationid = l.locationid;

-- fact_purchases
DROP VIEW IF EXISTS analytics.fact_purchases CASCADE;
CREATE VIEW analytics.fact_purchases AS
SELECT 
    poh.purchaseorderid,
    pod.purchaseorderdetailid,
    poh.vendorid,
    poh.orderdate,
    poh.shipdate,
    poh.status,
    pod.productid,
    pod.orderqty,
    pod.unitprice,
    pod.receivedqty,
    pod.rejectedqty,
    (pod.receivedqty - pod.rejectedqty) AS stockedqty,
    (pod.orderqty * pod.unitprice) AS linetotal
FROM purchasing.purchaseorderheader poh
INNER JOIN purchasing.purchaseorderdetail pod ON poh.purchaseorderid = pod.purchaseorderid;


-- ==========================================
-- TASK 1 & 2: Analytics Layer & Chained SQL Pipeline
-- LAYER 3: Analytical Aggregations 
-- ==========================================

-- customer_metrics
DROP VIEW IF EXISTS analytics.customer_metrics CASCADE;
CREATE VIEW analytics.customer_metrics AS
SELECT 
    c.customerid,
    c.customer_name,
    c.persontype,
    MIN(fs.orderdate) AS first_purchase_date,
    MAX(fs.orderdate) AS last_purchase_date,
    COUNT(DISTINCT fs.salesorderid) AS total_orders,
    SUM(fs.linetotal) AS customer_lifetime_value,
    CASE 
        WHEN SUM(fs.linetotal) > 10000 THEN 'High Value'
        WHEN SUM(fs.linetotal) > 2000 THEN 'Medium Value'
        ELSE 'Low Value'
    END AS customer_segment
FROM analytics.dim_customer c
INNER JOIN analytics.fact_sales fs ON c.customerid = fs.customerid
GROUP BY c.customerid, c.customer_name, c.persontype;

-- product_metrics
DROP VIEW IF EXISTS analytics.product_metrics CASCADE;
CREATE VIEW analytics.product_metrics AS
SELECT 
    p.productid,
    p.product_name,
    p.category_name,
    p.subcategory_name,
    SUM(fs.orderqty) AS total_quantity_sold,
    SUM(fs.linetotal) AS total_revenue,
    SUM(fs.profit) AS total_profit,
    COALESCE((SELECT SUM(quantity) FROM analytics.fact_inventory fi WHERE fi.productid = p.productid), 0) AS current_inventory
FROM analytics.dim_product p
LEFT JOIN analytics.fact_sales fs ON p.productid = fs.productid
GROUP BY p.productid, p.product_name, p.category_name, p.subcategory_name;



-- employee_performance
DROP VIEW IF EXISTS analytics.employee_performance CASCADE;
CREATE VIEW analytics.employee_performance AS
SELECT 
    e.employeeid,
    e.employee_name,
    e.jobtitle,
    e.salesquota,
    COUNT(DISTINCT fs.salesorderid) AS orders_handled,
    SUM(fs.linetotal) AS total_revenue_generated,
    SUM(fs.profit) AS total_profit_generated
FROM analytics.dim_employee e
INNER JOIN analytics.fact_sales fs ON e.employeeid = fs.salespersonid
GROUP BY e.employeeid, e.employee_name, e.jobtitle, e.salesquota;

-- regional_analysis
DROP VIEW IF EXISTS analytics.regional_analysis CASCADE;
CREATE VIEW analytics.regional_analysis AS
SELECT 
    t.territoryid,
    t.territory_name,
    t.country_name,
    t.territory_group,
    d.year,
    d.month,
    SUM(fs.linetotal) AS monthly_revenue,
    SUM(fs.profit) AS monthly_profit,
    COUNT(DISTINCT fs.salesorderid) AS order_count
FROM analytics.fact_sales fs
INNER JOIN analytics.dim_territory t ON fs.territoryid = t.territoryid
INNER JOIN analytics.dim_date d ON fs.orderdate::date = d.date
GROUP BY t.territoryid, t.territory_name, t.country_name, t.territory_group, d.year, d.month;


-- ==========================================
-- TASK 3: Executive KPI Datasets 
-- TASK 4: Advanced SQL 

-- LAYER 4: Final Executive KPIs
-- ==========================================

-- kpi_sales 1
DROP VIEW IF EXISTS analytics.kpi_sales CASCADE;
CREATE VIEW analytics.kpi_sales AS
WITH monthly_sales AS (
    SELECT 
        d.year,
        d.month,
        SUM(fs.linetotal) AS total_revenue,
        SUM(CASE WHEN fs.onlineorderflag = true THEN fs.linetotal ELSE 0 END) AS online_revenue,
        SUM(CASE WHEN fs.onlineorderflag = false THEN fs.linetotal ELSE 0 END) AS offline_revenue
    FROM analytics.fact_sales fs
    INNER JOIN analytics.dim_date d ON fs.orderdate::date = d.date
    GROUP BY d.year, d.month
),
sales_with_lag AS (
    SELECT 
        year,
        month,
        total_revenue,
        online_revenue,
        offline_revenue,
        LAG(total_revenue, 1) OVER (ORDER BY year, month) AS prev_month_revenue
    FROM monthly_sales
)
SELECT 
    year,
    month,
    total_revenue,
    online_revenue,
    offline_revenue,
    prev_month_revenue,
    CASE 
        WHEN prev_month_revenue IS NULL OR prev_month_revenue = 0 THEN 0
        ELSE ((total_revenue - prev_month_revenue) / prev_month_revenue) * 100 
    END AS mom_growth_pct
FROM sales_with_lag;

-- kpi_customers (Segments, Retention)
DROP VIEW IF EXISTS analytics.kpi_customers CASCADE;
CREATE VIEW analytics.kpi_customers AS
SELECT 
    customer_segment,
    COUNT(customerid) AS customer_count,
    SUM(customer_lifetime_value) AS total_segment_value,
    AVG(customer_lifetime_value) AS avg_customer_value,
    SUM(CASE WHEN total_orders > 1 THEN 1 ELSE 0 END) AS repeat_customers,
    (SUM(CASE WHEN total_orders > 1 THEN 1 ELSE 0 END)::numeric / NULLIF(COUNT(customerid), 0)) * 100 AS retention_rate_pct
FROM analytics.customer_metrics
GROUP BY customer_segment;

-- kpi_products (Best/Worst Selling, Profitability, Category Perf)
DROP VIEW IF EXISTS analytics.kpi_products CASCADE;
CREATE VIEW analytics.kpi_products AS
SELECT 
    productid,
    product_name,
    category_name,
    total_revenue,
    total_profit,
    DENSE_RANK() OVER (ORDER BY total_revenue DESC NULLS LAST) AS sales_rank,
    DENSE_RANK() OVER (PARTITION BY category_name ORDER BY total_profit DESC NULLS LAST) AS category_profit_rank
FROM analytics.product_metrics
WHERE total_revenue > 0;

-- kpi_employees (Rankings, Revenue Contribution)
DROP VIEW IF EXISTS analytics.kpi_employees CASCADE;
CREATE VIEW analytics.kpi_employees AS
SELECT 
    employeeid,
    employee_name,
    total_revenue_generated,
    salesquota,
    CASE 
        WHEN salesquota > 0 THEN (total_revenue_generated / salesquota) * 100 
        ELSE NULL 
    END AS quota_achievement_pct,
    RANK() OVER (ORDER BY total_revenue_generated DESC) AS overall_sales_rank
FROM analytics.employee_performance;

-- kpi_territories (Regional Revenue, Top/Bottom Territories)
DROP VIEW IF EXISTS analytics.kpi_territories CASCADE;
CREATE VIEW analytics.kpi_territories AS
WITH territory_totals AS (
    SELECT 
        territory_name,
        country_name,
        territory_group,
        SUM(monthly_revenue) AS total_revenue,
        SUM(monthly_profit) AS total_profit
    FROM analytics.regional_analysis
    GROUP BY territory_name, country_name, territory_group
)
SELECT 
    territory_name,
    country_name,
    territory_group,
    total_revenue,
    total_profit,
    RANK() OVER (ORDER BY total_revenue DESC) AS territory_rank,
    CASE 
        WHEN RANK() OVER (ORDER BY total_revenue DESC) <= 3 THEN 'Top Performer'
        WHEN RANK() OVER (ORDER BY total_revenue ASC) <= 3 THEN 'Bottom Performer'
        ELSE 'Average'
    END AS performance_status
FROM territory_totals;

-- kpi_inventory (Health, Low Stock, Purchasing Trends)
DROP VIEW IF EXISTS analytics.kpi_inventory CASCADE;
CREATE VIEW analytics.kpi_inventory AS
SELECT 
    pm.productid,
    pm.product_name,
    pm.category_name,
    p.safetystocklevel,
    p.reorderpoint,
    pm.current_inventory,
    CASE 
        WHEN pm.current_inventory <= p.reorderpoint THEN 'Low Stock - Reorder Needed'
        WHEN pm.current_inventory > (p.safetystocklevel * 2) THEN 'Overstocked'
        ELSE 'Healthy'
    END AS inventory_health_status
FROM analytics.product_metrics pm
INNER JOIN production.product p ON pm.productid = p.productid;
