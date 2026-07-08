-- Part 1 — Aggregation Basics

-- 1. Find the total revenue generated per store.
SELECT 
    s.store_id, 
    SUM(p.amount) AS total_revenue
FROM store s
JOIN staff st ON s.store_id = st.store_id
JOIN payment p ON st.staff_id = p.staff_id
GROUP BY s.store_id;

-- 2. Find the average rental duration per film category.
SELECT 
    c.name AS category_name, 
    ROUND(AVG(f.rental_duration), 2) AS avg_rental_duration
FROM category c
JOIN film_category fc ON c.category_id = fc.category_id
JOIN film f ON fc.film_id = f.film_id
GROUP BY c.category_id, c.name;

-- 3. Find the number of rentals made each month.
SELECT 
    TO_CHAR(rental_date, 'YYYY-MM') AS rental_month,
    COUNT(rental_id) AS total_rentals
FROM rental
GROUP BY TO_CHAR(rental_date, 'YYYY-MM')
ORDER BY rental_month;

-- 4. Find categories with more than 50 films (use HAVING).
SELECT 
    c.name AS category_name, 
    COUNT(fc.film_id) AS film_count
FROM category c
JOIN film_category fc ON c.category_id = fc.category_id
GROUP BY c.category_id, c.name
HAVING COUNT(fc.film_id) > 50
ORDER BY film_count DESC;


-- Part 2 — Subquery Challenges

-- 5. Find customers who spent more than the average customer spend.
SELECT 
    c.customer_id, 
    c.first_name, 
    c.last_name, 
    SUM(p.amount) AS total_spent
FROM customer c
JOIN payment p ON c.customer_id = p.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name
HAVING SUM(p.amount) > (
    SELECT SUM(amount) / COUNT(DISTINCT customer_id) FROM payment
)
ORDER BY total_spent DESC;

-- 6. Find the film(s) with the highest rental rate in each category (use a correlated subquery).
SELECT 
    c.name AS category_name, 
    f.title, 
    f.rental_rate
FROM category c
JOIN film_category fc ON c.category_id = fc.category_id
JOIN film f ON fc.film_id = f.film_id
WHERE f.rental_rate = (
    SELECT MAX(f2.rental_rate)
    FROM film f2
    JOIN film_category fc2 ON f2.film_id = fc2.film_id
    WHERE fc2.category_id = c.category_id
)
ORDER BY c.name, f.title;

-- 7. Find customers who have never rented a film (use NOT IN / NOT EXISTS).
SELECT 
    customer_id, 
    first_name, 
    last_name
FROM customer c
WHERE NOT EXISTS (
    SELECT 1 FROM rental r WHERE r.customer_id = c.customer_id
);

-- 8. Find the store with the highest total revenue using a subquery in the WHERE clause.
SELECT 
    store_id, 
    total_revenue
FROM (
    SELECT 
        st.store_id, 
        SUM(p.amount) AS total_revenue
    FROM staff st
    JOIN payment p ON st.staff_id = p.staff_id
    GROUP BY st.store_id
) store_revenues
WHERE total_revenue = (
    SELECT MAX(revenue)
    FROM (
        SELECT SUM(amount) AS revenue 
        FROM staff st2
        JOIN payment p2 ON st2.staff_id = p2.staff_id
        GROUP BY st2.store_id
    ) max_rev
);


-- Part 3 — CTE & Window Function Challenges

-- 9. Using a CTE, rank customers by total spend within each city.
WITH CustomerCitySpend AS (
    SELECT 
        ci.city,
        c.customer_id,
        c.first_name,
        c.last_name,
        SUM(p.amount) AS total_spent
    FROM customer c
    JOIN address a ON c.address_id = a.address_id
    JOIN city ci ON a.city_id = ci.city_id
    JOIN payment p ON c.customer_id = p.customer_id
    GROUP BY ci.city, c.customer_id, c.first_name, c.last_name
)
SELECT 
    city,
    first_name,
    last_name,
    total_spent,
    RANK() OVER(PARTITION BY city ORDER BY total_spent DESC) AS spend_rank
FROM CustomerCitySpend
ORDER BY city, spend_rank;

-- 10. Using ROW_NUMBER(), find the most recently rented film for each customer.
WITH RankedRentals AS (
    SELECT 
        c.customer_id,
        c.first_name,
        c.last_name,
        f.title,
        r.rental_date,
        ROW_NUMBER() OVER(PARTITION BY c.customer_id ORDER BY r.rental_date DESC) as rn
    FROM customer c
    JOIN rental r ON c.customer_id = r.customer_id
    JOIN inventory i ON r.inventory_id = i.inventory_id
    JOIN film f ON i.film_id = f.film_id
)
SELECT 
    customer_id,
    first_name,
    last_name,
    title,
    rental_date
FROM RankedRentals
WHERE rn = 1;

-- 11. Using a CTE, calculate month-over-month rental revenue growth.
WITH MonthlyRevenue AS (
    SELECT 
        TO_CHAR(payment_date, 'YYYY-MM') AS month,
        SUM(amount) AS revenue
    FROM payment
    GROUP BY TO_CHAR(payment_date, 'YYYY-MM')
)
SELECT 
    month,
    revenue,
    LAG(revenue) OVER(ORDER BY month) AS prev_month_revenue,
    ROUND(((revenue - LAG(revenue) OVER(ORDER BY month)) / LAG(revenue) OVER(ORDER BY month)) * 100, 2) AS growth_percentage
FROM MonthlyRevenue
ORDER BY month;

-- 12. Find the top 3 highest-grossing films per category using RANK() inside a CTE.
WITH FilmRevenue AS (
    SELECT 
        c.name AS category_name,
        f.title,
        SUM(p.amount) AS revenue
    FROM category c
    JOIN film_category fc ON c.category_id = fc.category_id
    JOIN film f ON fc.film_id = f.film_id
    JOIN inventory i ON f.film_id = i.film_id
    JOIN rental r ON i.inventory_id = r.inventory_id
    JOIN payment p ON r.rental_id = p.rental_id
    GROUP BY c.category_id, c.name, f.film_id, f.title
),
RankedFilms AS (
    SELECT 
        category_name,
        title,
        revenue,
        RANK() OVER(PARTITION BY category_name ORDER BY revenue DESC) as rank
    FROM FilmRevenue
)
SELECT 
    category_name,
    title,
    revenue,
    rank
FROM RankedFilms
WHERE rank <= 3;


-- Bonus Challenge
WITH StaffRevenue AS (
    SELECT 
        s.store_id,
        st.staff_id,
        st.first_name || ' ' || st.last_name AS staff_name,
        SUM(p.amount) AS total_processed
    FROM store s
    JOIN staff st ON s.store_id = st.store_id
    JOIN payment p ON st.staff_id = p.staff_id
    GROUP BY s.store_id, st.staff_id, staff_name
),
StoreRevenue AS (
    SELECT 
        store_id,
        SUM(total_processed) AS store_total
    FROM StaffRevenue
    GROUP BY store_id
),
RankedStaff AS (
    SELECT 
        sr.store_id,
        sr.staff_name,
        sr.total_processed,
        str.store_total,
        ROUND((sr.total_processed / str.store_total) * 100, 2) AS contribution_percentage,
        RANK() OVER(PARTITION BY sr.store_id ORDER BY sr.total_processed DESC) as rank
    FROM StaffRevenue sr
    JOIN StoreRevenue str ON sr.store_id = str.store_id
)
SELECT 
    store_id,
    staff_name,
    total_processed,
    contribution_percentage
FROM RankedStaff
WHERE rank = 1;
