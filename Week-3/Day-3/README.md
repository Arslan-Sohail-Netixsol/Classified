# Week 3 Day 3: Aggregation, Subqueries, CTEs, and Window Functions

## SQL Concepts Explained

### Subqueries vs. CTEs vs. Window Functions
- **Subquery**: A query nested inside another query (e.g., inside `SELECT`, `FROM`, `WHERE`, or `HAVING`). Use a subquery for quick, one-off calculations that filter data (like finding a specific max value to compare against). Correlated subqueries are evaluated row-by-row against the outer query.
- **CTE (Common Table Expression)**: Defined using the `WITH` clause, it acts as a temporary result set that you can reference within a `SELECT` statement. Use a CTE when a subquery becomes too complex, when you need to reference the same subquery multiple times, or to make your code much more readable by breaking it down into logical steps.
- **Window Function**: Performs a calculation across a set of table rows that are somehow related to the current row (e.g., using `OVER()`, `PARTITION BY`, `ORDER BY`). Unlike `GROUP BY` aggregation which collapses rows into a single summary row, window functions preserve the original rows while adding the calculated column. Use window functions for ranking, running totals, moving averages, and accessing previous/next rows (`LAG`/`LEAD`).

## Part 1: Aggregation Basics Solutions

**1. Find the total revenue generated per store.**
- *Solution*: Joined `store` to `staff`, and `staff` to `payment`. Grouped by `store_id` and used the `SUM()` aggregate function on `amount`. 

**2. Find the average rental duration per film category.**
- *Solution*: Joined `category` to `film_category` to `film`. Grouped by category and used `ROUND(AVG(rental_duration), 2)` to get clean decimal values.

**3. Find the number of rentals made each month.**
- *Solution*: Used `TO_CHAR(rental_date, 'YYYY-MM')` to extract the year and month from the rental timestamp. Grouped by this formatted date and counted `rental_id`.

**4. Find categories with more than 50 films.**
- *Solution*: Joined `category` to `film_category`, grouped by category, and used the `HAVING` clause to filter the results post-aggregation (`HAVING COUNT(film_id) > 50`).

## Part 2: Subquery Challenges Solutions

**5. Find customers who spent more than the average customer spend.**
- *Solution*: Calculated the total spent per customer, then used a `HAVING` clause containing a subquery that calculates the global average spend (`SUM(amount) / COUNT(DISTINCT customer_id)`).

**6. Find the film(s) with the highest rental rate in each category.**
- *Solution*: Used a *correlated subquery* in the `WHERE` clause. For each row in the outer query, the subquery finds the maximum rental rate specifically for that row's `category_id`.

**7. Find customers who have never rented a film.**
- *Solution*: Used `NOT EXISTS` with a subquery checking the `rental` table for the outer query's `customer_id`. 

**8. Find the store with the highest total revenue.**
- *Solution*: Calculated the total revenue per store in a subquery in the `FROM` clause. Then, filtered it in the `WHERE` clause using another subquery to find the maximum revenue among all stores.

## Part 3: CTE & Window Function Challenges Solutions

**9. Rank customers by total spend within each city.**
- *Solution*: Used a CTE to calculate total spend per customer per city. In the main query, used `RANK() OVER(PARTITION BY city ORDER BY total_spent DESC)` to reset the rank for each new city.

**10. Most recently rented film for each customer.**
- *Solution*: Created a CTE using `ROW_NUMBER() OVER(PARTITION BY customer_id ORDER BY rental_date DESC)`. This gives the latest rental a row number of `1`. The main query then filtered for `WHERE rn = 1`.

**11. Month-over-month rental revenue growth.**
- *Solution*: Created a CTE calculating total revenue per month. Used the `LAG()` window function to look at the previous month's revenue and calculated the percentage change.

**12. Top 3 highest-grossing films per category.**
- *Solution*: Built a chain of two CTEs. First CTE calculated total revenue per film per category. Second CTE ranked them using `RANK() OVER(PARTITION BY category_name)`. Main query filtered for `rank <= 3`.

## :star: Bonus Challenge Solution
**Which staff member processed the highest revenue in each store, and what percentage of that store's total revenue did they contribute?**
- *Solution*: 
    1. First CTE (`StaffRevenue`): Calculated total revenue processed by each staff member.
    2. Second CTE (`StoreRevenue`): Calculated the overall total revenue for each store.
    3. Third CTE (`RankedStaff`): Joined the first two CTEs together to calculate the percentage (`(total_processed / store_total) * 100`). Applied `RANK() OVER(PARTITION BY store_id)` to rank the staff members.
    4. Main Query: Filtered for `rank = 1` to get the top performer per store.
