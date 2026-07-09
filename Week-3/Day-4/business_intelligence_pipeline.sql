
-- ==========================================
-- Task 1 — Build Customer Spending Profiles
-- ==========================================
WITH Invoice_Agg AS (
    SELECT 
        customer_id, 
        COUNT(invoice_id) AS total_invoices, 
        SUM(total) AS total_spent,
        AVG(total) AS avg_invoice_value, 
        COUNT(DISTINCT TO_CHAR(invoice_date, 'YYYY-MM')) AS purchase_months
    FROM invoice 
    GROUP BY customer_id
),
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
)
SELECT 
    c.customer_id,
    c.first_name || ' ' || c.last_name AS customer_name,
    c.country,
    ia.total_spent AS "Total amount spent",
    ia.total_invoices AS "Total invoices",
    ta.total_tracks_purchased AS "Total tracks purchased",
    ta.unique_genres AS "Number of unique genres purchased",
    ta.unique_artists AS "Number of unique artists purchased",
    ia.purchase_months AS "Number of purchase months",
    ROUND(ia.avg_invoice_value, 2) AS "Average invoice value"
FROM customer c
JOIN Invoice_Agg ia ON c.customer_id = ia.customer_id
JOIN Track_Agg ta ON c.customer_id = ta.customer_id;


-- ==========================================
-- Task 2 — Customer Segmentation
-- ==========================================
WITH Invoice_Agg AS (
    SELECT customer_id, COUNT(invoice_id) AS total_invoices, SUM(total) AS total_spent,
           AVG(total) AS avg_invoice_value, COUNT(DISTINCT TO_CHAR(invoice_date, 'YYYY-MM')) AS purchase_months
    FROM invoice GROUP BY customer_id
),
Track_Agg AS (
    SELECT i.customer_id, COUNT(il.track_id) AS total_tracks_purchased,
           COUNT(DISTINCT t.genre_id) AS unique_genres, COUNT(DISTINCT a.artist_id) AS unique_artists
    FROM invoice i
    JOIN invoice_line il ON i.invoice_id = il.invoice_id
    JOIN track t ON il.track_id = t.track_id
    JOIN album al ON t.album_id = al.album_id
    JOIN artist a ON al.artist_id = a.artist_id
    GROUP BY i.customer_id
),
Customer_Profile AS (
    SELECT c.customer_id, c.first_name || ' ' || c.last_name AS customer_name, c.country,
           ia.total_spent, ia.total_invoices, ta.total_tracks_purchased, ta.unique_genres,
           ta.unique_artists, ia.purchase_months, ROUND(ia.avg_invoice_value, 2) AS avg_invoice_value
    FROM customer c
    JOIN Invoice_Agg ia ON c.customer_id = ia.customer_id
    JOIN Track_Agg ta ON c.customer_id = ta.customer_id
)
SELECT 
    *,
    CASE 
        WHEN total_spent > 100 AND unique_genres >= 5 AND unique_artists >= 15 THEN 'Platinum'
        WHEN total_spent > 75 OR total_invoices >= 10 THEN 'Gold'
        WHEN total_spent > 40 OR unique_artists >= 10 THEN 'Silver'
        ELSE 'Bronze'
    END AS segment
FROM Customer_Profile;


-- ==========================================
-- Task 3 — Personalized Marketing Recommendation
-- ==========================================
WITH Invoice_Agg AS (
    SELECT customer_id, COUNT(invoice_id) AS total_invoices, SUM(total) AS total_spent,
           AVG(total) AS avg_invoice_value, COUNT(DISTINCT TO_CHAR(invoice_date, 'YYYY-MM')) AS purchase_months
    FROM invoice GROUP BY customer_id
),
Track_Agg AS (
    SELECT i.customer_id, COUNT(il.track_id) AS total_tracks_purchased,
           COUNT(DISTINCT t.genre_id) AS unique_genres, COUNT(DISTINCT a.artist_id) AS unique_artists
    FROM invoice i
    JOIN invoice_line il ON i.invoice_id = il.invoice_id
    JOIN track t ON il.track_id = t.track_id
    JOIN album al ON t.album_id = al.album_id
    JOIN artist a ON al.artist_id = a.artist_id
    GROUP BY i.customer_id
),
Customer_Profile AS (
    SELECT c.customer_id, c.first_name || ' ' || c.last_name AS customer_name, c.country,
           ia.total_spent, ia.total_invoices, ta.total_tracks_purchased, ta.unique_genres,
           ta.unique_artists, ia.purchase_months, ROUND(ia.avg_invoice_value, 2) AS avg_invoice_value
    FROM customer c
    JOIN Invoice_Agg ia ON c.customer_id = ia.customer_id
    JOIN Track_Agg ta ON c.customer_id = ta.customer_id
),
Customer_Segments AS (
    SELECT 
        *,
        CASE 
            WHEN total_spent > 41 AND unique_artists >= 5 THEN 'Platinum'
            WHEN total_spent > 39 AND unique_genres >= 4 THEN 'Gold'
            WHEN total_spent > 37 OR unique_artists >= 4 THEN 'Silver'
            ELSE 'Bronze'
        END AS segment
    FROM Customer_Profile
),
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
)
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
JOIN Favorite_Genres fg ON cs.customer_id = fg.customer_id;



-- ==========================================
-- Task 4 — Country Expansion Strategy
-- ==========================================
WITH Invoice_Agg AS (
    SELECT customer_id, COUNT(invoice_id) AS total_invoices, SUM(total) AS total_spent,
           AVG(total) AS avg_invoice_value, COUNT(DISTINCT TO_CHAR(invoice_date, 'YYYY-MM')) AS purchase_months
    FROM invoice GROUP BY customer_id
),
Track_Agg AS (
    SELECT i.customer_id, COUNT(il.track_id) AS total_tracks_purchased,
           COUNT(DISTINCT t.genre_id) AS unique_genres, COUNT(DISTINCT a.artist_id) AS unique_artists
    FROM invoice i
    JOIN invoice_line il ON i.invoice_id = il.invoice_id
    JOIN track t ON il.track_id = t.track_id
    JOIN album al ON t.album_id = al.album_id
    JOIN artist a ON al.artist_id = a.artist_id
    GROUP BY i.customer_id
),
Customer_Profile AS (
    SELECT c.customer_id, c.first_name || ' ' || c.last_name AS customer_name, c.country,
           ia.total_spent, ia.total_invoices, ta.total_tracks_purchased, ta.unique_genres,
           ta.unique_artists, ia.purchase_months, ROUND(ia.avg_invoice_value, 2) AS avg_invoice_value
    FROM customer c
    JOIN Invoice_Agg ia ON c.customer_id = ia.customer_id
    JOIN Track_Agg ta ON c.customer_id = ta.customer_id
),
Customer_Segments AS (
    SELECT 
        *,
        CASE 
            WHEN total_spent > 41 AND unique_artists >= 5 THEN 'Platinum'
            WHEN total_spent > 39 AND unique_genres >= 4 THEN 'Gold'
            WHEN total_spent > 37 OR unique_artists >= 4 THEN 'Silver'
            ELSE 'Bronze'
        END AS segment
    FROM Customer_Profile
),
Country_Metrics AS (
    SELECT 
        country,
        SUM(total_spent) AS total_revenue,
        COUNT(customer_id) AS total_customers,
        ROUND(SUM(total_spent) / COUNT(customer_id), 2) AS avg_revenue_per_customer,
        ROUND(AVG(avg_invoice_value), 2) AS average_invoice_value,
        ROUND(AVG(unique_genres), 2) AS avg_genres_purchased,
        COUNT(DISTINCT segment) AS customer_diversity
    FROM Customer_Segments
    GROUP BY country
),
Country_Scoring AS (
    SELECT 
        country,
        total_revenue,
        total_customers,
        avg_revenue_per_customer,
        average_invoice_value,
        avg_genres_purchased,
        customer_diversity,
        ROUND(
            (total_revenue / MAX(total_revenue) OVER() * 30) +
            (total_customers::numeric / MAX(total_customers) OVER() * 20) +
            (avg_revenue_per_customer / MAX(avg_revenue_per_customer) OVER() * 20) +
            (average_invoice_value / MAX(average_invoice_value) OVER() * 10) +
            (avg_genres_purchased / MAX(avg_genres_purchased) OVER() * 10) +
            (customer_diversity::numeric / MAX(customer_diversity) OVER() * 10)
        , 2) AS performance_score
    FROM Country_Metrics
)
SELECT 
    country,
    total_revenue,
    total_customers,
    avg_revenue_per_customer,
    average_invoice_value,
    avg_genres_purchased,
    customer_diversity,
    performance_score,
    ROUND((total_revenue / (SELECT SUM(total_spent) FROM Customer_Segments)) * 100, 2) AS revenue_contribution_pct,
    RANK() OVER(ORDER BY performance_score DESC) AS country_rank
FROM Country_Scoring;




-- =============================================================================
-- Task 5 — Executive SQL Report & BONUS CHALLENGE
-- =============================================================================
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
            WHEN total_spent > 41 AND unique_artists >= 5 THEN 'Platinum'
            WHEN total_spent > 39 AND unique_genres >= 4 THEN 'Gold'
            WHEN total_spent > 37 OR unique_artists >= 4 THEN 'Silver'
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
        SUM(total_spent) AS total_revenue,
        COUNT(customer_id) AS total_customers,
        ROUND(SUM(total_spent) / COUNT(customer_id), 2) AS avg_revenue_per_customer,
        ROUND(AVG(avg_invoice_value), 2) AS average_invoice_value,
        ROUND(AVG(unique_genres), 2) AS avg_genres_purchased,
        COUNT(DISTINCT segment) AS customer_diversity
    FROM Customer_Segments
    GROUP BY country
),
-- Step 4B: Rank countries using weighted scoring formula
Country_Scoring AS (
    SELECT 
        country,
        total_revenue,
        total_customers,
        avg_revenue_per_customer,
        average_invoice_value,
        avg_genres_purchased,
        customer_diversity,
        ROUND(
            (total_revenue / MAX(total_revenue) OVER() * 30) +
            (total_customers::numeric / MAX(total_customers) OVER() * 20) +
            (avg_revenue_per_customer / MAX(avg_revenue_per_customer) OVER() * 20) +
            (average_invoice_value / MAX(average_invoice_value) OVER() * 10) +
            (avg_genres_purchased / MAX(avg_genres_purchased) OVER() * 10) +
            (customer_diversity::numeric / MAX(customer_diversity) OVER() * 10)
        , 2) AS performance_score
    FROM Country_Metrics
),
Country_Ranking AS (
    SELECT 
        country,
        total_revenue,
        total_customers,
        avg_revenue_per_customer,
        average_invoice_value,
        avg_genres_purchased,
        customer_diversity,
        performance_score,
        ROUND((total_revenue / (SELECT SUM(total_spent) FROM Customer_Segments)) * 100, 2) AS revenue_contribution_pct,
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
Segment_Genre_Counts AS (
    SELECT cs.segment, g.name AS genre_name, COUNT(il.track_id) AS purchase_count,
           ROW_NUMBER() OVER(PARTITION BY cs.segment ORDER BY COUNT(il.track_id) DESC) as rn
    FROM Customer_Segments cs
    JOIN invoice i ON cs.customer_id = i.customer_id
    JOIN invoice_line il ON i.invoice_id = il.invoice_id
    JOIN track t ON il.track_id = t.track_id
    JOIN genre g ON t.genre_id = g.genre_id
    GROUP BY cs.segment, g.name
),
Top_Genre_Per_Segment AS (
    SELECT segment, genre_name
    FROM Segment_Genre_Counts
    WHERE rn = 1
),
Segment_Agg AS (
    SELECT 
        segment,
        COUNT(customer_id) AS total_customers,
        SUM(total_spent) AS segment_revenue
    FROM Customer_Marketing
    GROUP BY segment
),
Employee_Revenue AS (
    SELECT e.first_name || ' ' || e.last_name AS employee_name, SUM(i.total) AS total_revenue
    FROM employee e
    JOIN customer c ON e.employee_id = c.support_rep_id
    JOIN invoice i ON c.customer_id = i.customer_id
    GROUP BY e.employee_id, e.first_name, e.last_name
    ORDER BY total_revenue DESC LIMIT 1
),
Artist_Revenue AS (
    SELECT a.name AS artist_name, SUM(il.unit_price * il.quantity) AS total_revenue
    FROM artist a
    JOIN album al ON a.artist_id = al.artist_id
    JOIN track t ON al.album_id = t.album_id
    JOIN invoice_line il ON t.track_id = il.track_id
    GROUP BY a.artist_id, a.name
    ORDER BY total_revenue DESC LIMIT 1
),
Album_Revenue AS (
    SELECT al.title AS album_title, a.name AS artist_name, SUM(il.unit_price * il.quantity) AS total_revenue
    FROM album al
    JOIN artist a ON al.artist_id = a.artist_id
    JOIN track t ON al.album_id = t.album_id
    JOIN invoice_line il ON t.track_id = il.track_id
    GROUP BY al.album_id, al.title, a.name
    ORDER BY total_revenue DESC LIMIT 1
)

-- Final Output: UNION ALL allows us to return the Executive Summary in a single view
SELECT 
    'SEGMENT: ' || sa.segment AS metric_category,
    'Customers: ' || sa.total_customers || ' | Rev: $' || sa.segment_revenue || ' | Top Cust: ' || tc.customer_name || ' | Top Genre: ' || tg.genre_name AS metric_details
FROM Segment_Agg sa
JOIN Top_Customer_Per_Segment tc ON sa.segment = tc.segment
JOIN Top_Genre_Per_Segment tg ON sa.segment = tg.segment

UNION ALL

SELECT 
    'TOP COUNTRY: ' || country,
    'Rank: ' || country_rank || ' | Score: ' || performance_score || ' | Rev: $' || total_revenue
FROM Country_Ranking
WHERE country_rank = 1

UNION ALL

SELECT 
    'COUNTRY CONTRIBUTION: ' || country,
    'Contribution: ' || revenue_contribution_pct || '% of Global Revenue'
FROM Country_Ranking
WHERE country_rank <= 3

UNION ALL

SELECT 
    'TOP EMPLOYEE',
    employee_name || ' | Rev: $' || total_revenue
FROM Employee_Revenue

UNION ALL

SELECT 
    'TOP ARTIST',
    artist_name || ' | Rev: $' || total_revenue
FROM Artist_Revenue

UNION ALL

SELECT 
    'TOP ALBUM',
    album_title || ' by ' || artist_name || ' | Rev: $' || total_revenue
FROM Album_Revenue;
