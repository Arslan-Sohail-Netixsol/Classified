/*
=============================================================================
  Music Store Business Intelligence Pipeline
  Tasks 1-5 + Bonus Challenge
  Description: A single, chained CTE pipeline that builds customer profiles, 
  segments them, analyzes countries, and outputs an executive summary.
=============================================================================
*/

WITH 
-- ==========================================
-- TASK 1: Build Customer Spending Profiles
-- ==========================================
-- Step 1A: Aggregate Invoice-level metrics 
Invoice_Agg AS (
    SELECT 
        customer_id,
        COUNT(invoice_id) AS total_invoices,
        SUM(total) AS total_spent,
        AVG(total) AS avg_invoice_value,
        COUNT(DISTINCT TO_CHAR(invoice_date, 'YYYY-MM')) AS purchase_months
    FROM invoice
    GROUP BY customer_id
),
-- Step 1B: Aggregate Track/Line-level metrics
Track_Agg AS (
    SELECT 
        i.customer_id,
        COUNT(il.track_id) AS total_tracks_purchased,
        COUNT(DISTINCT t.genre_id) AS unique_genres,
        COUNT(DISTINCT a.artist_id) AS unique_artists
    FROM invoice i
    JOIN invoice_line il ON i.invoice_id = il.invoice_id
    JOIN track t ON il.track_id = t.track_id
    JOIN album al ON t.album_id = al.album_id
    JOIN artist a ON al.artist_id = a.artist_id
    GROUP BY i.customer_id
),
-- Step 1C: Combine into the final Profile
Customer_Profile AS (
    SELECT 
        c.customer_id,
        c.first_name || ' ' || c.last_name AS customer_name,
        c.country,
        ia.total_spent,
        ia.total_invoices,
        ta.total_tracks_purchased,
        ta.unique_genres,
        ta.unique_artists,
        ia.purchase_months,
        ROUND(ia.avg_invoice_value, 2) AS avg_invoice_value
    FROM customer c
    JOIN Invoice_Agg ia ON c.customer_id = ia.customer_id
    JOIN Track_Agg ta ON c.customer_id = ta.customer_id
),

-- ==========================================
-- TASK 2: Customer Segmentation
-- ==========================================
Customer_Segments AS (
    SELECT 
        *,
        CASE 
            WHEN total_spent > 100 AND unique_genres >= 5 THEN 'Platinum'
            WHEN total_spent > 75 OR total_invoices >= 10 THEN 'Gold'
            WHEN total_spent > 40 THEN 'Silver'
            ELSE 'Bronze'
        END AS segment
    FROM Customer_Profile
),

-- ==========================================
-- TASK 3: Personalized Marketing Recommendation
-- ==========================================
-- Step 3A: Find favorite genre via Window Function
Genre_Counts AS (
    SELECT 
        i.customer_id,
        g.name AS genre_name,
        COUNT(il.track_id) AS purchase_count,
        ROW_NUMBER() OVER(PARTITION BY i.customer_id ORDER BY COUNT(il.track_id) DESC) as rn
    FROM invoice i
    JOIN invoice_line il ON i.invoice_id = il.invoice_id
    JOIN track t ON il.track_id = t.track_id
    JOIN genre g ON t.genre_id = g.genre_id
    GROUP BY i.customer_id, g.name
),
Favorite_Genres AS (
    SELECT customer_id, genre_name AS favorite_genre
    FROM Genre_Counts
    WHERE rn = 1
),
-- Step 3B: Assign Marketing Campaign
Customer_Marketing AS (
    SELECT 
        cs.customer_id,
        cs.customer_name,
        cs.country,
        cs.total_spent,
        cs.segment,
        fg.favorite_genre,
        CASE 
            WHEN cs.segment = 'Platinum' THEN 'Early access to new releases in ' || fg.favorite_genre
            WHEN cs.segment = 'Gold' THEN 'Exclusive Album Bundles in ' || fg.favorite_genre
            WHEN cs.segment = 'Silver' THEN '15% Off all ' || fg.favorite_genre || ' Tracks'
            WHEN cs.segment = 'Bronze' THEN 'First purchase coupon for ' || fg.favorite_genre
        END AS promotional_campaign
    FROM Customer_Segments cs
    JOIN Favorite_Genres fg ON cs.customer_id = fg.customer_id
),

-- ==========================================
-- TASK 4: Country Expansion Strategy
-- ==========================================
Country_Metrics AS (
    SELECT 
        country,
        SUM(total_spent) AS country_revenue,
        COUNT(customer_id) AS total_customers,
        ROUND(SUM(total_spent) / COUNT(customer_id), 2) AS avg_revenue_per_customer,
        ROUND(AVG(avg_invoice_value), 2) AS country_avg_invoice,
        MAX(unique_genres) AS max_genres_purchased
    FROM Customer_Segments
    GROUP BY country
),
-- Step 4B: Rank countries using weighted scoring formula
Country_Scoring AS (
    SELECT 
        country,
        country_revenue,
        total_customers,
        avg_revenue_per_customer,
        ROUND(
            (country_revenue / MAX(country_revenue) OVER() * 40) +
            (avg_revenue_per_customer / MAX(avg_revenue_per_customer) OVER() * 30) +
            (total_customers::numeric / MAX(total_customers) OVER() * 30)
        , 2) AS performance_score
    FROM Country_Metrics
),
Country_Ranking AS (
    SELECT 
        country,
        country_revenue,
        total_customers,
        avg_revenue_per_customer,
        performance_score,
        RANK() OVER(ORDER BY performance_score DESC) AS country_rank
    FROM Country_Scoring
),

-- ==========================================
-- TASK 5: Executive SQL Report (Final Output)
-- ==========================================
-- Prepare Segment level aggregates
Top_Customer_Per_Segment AS (
    SELECT segment, customer_name, favorite_genre, total_spent
    FROM (
        SELECT segment, customer_name, favorite_genre, total_spent,
               ROW_NUMBER() OVER(PARTITION BY segment ORDER BY total_spent DESC) as rn
        FROM Customer_Marketing
    ) x WHERE rn = 1
),
Segment_Agg AS (
    SELECT 
        segment,
        COUNT(customer_id) AS total_customers,
        SUM(total_spent) AS segment_revenue
    FROM Customer_Marketing
    GROUP BY segment
)

-- Final Output: UNION ALL allows us to return the Executive Summary in a single view
SELECT 
    'SEGMENT: ' || sa.segment AS metric_category,
    'Customers: ' || sa.total_customers || ' | Rev: $' || sa.segment_revenue || ' | Top Cust: ' || tc.customer_name AS metric_details
FROM Segment_Agg sa
JOIN Top_Customer_Per_Segment tc ON sa.segment = tc.segment

UNION ALL

SELECT 
    'TOP COUNTRY: ' || country,
    'Rank: ' || country_rank || ' | Score: ' || performance_score || ' | Rev: $' || country_revenue
FROM Country_Ranking
WHERE country_rank <= 3;
