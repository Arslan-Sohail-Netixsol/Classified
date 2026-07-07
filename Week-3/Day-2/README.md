# DVD Rental Database Analysis

## Part 1: Relationship Discovery

### Primary Keys
- **actor**: `actor_id`
- **film**: `film_id`
- **category**: `category_id`
- **customer**: `customer_id`
- **address**: `address_id`
- **city**: `city_id`
- **country**: `country_id`
- **inventory**: `inventory_id`
- **rental**: `rental_id`
- **payment**: `payment_id`
- **staff**: `staff_id`
- **store**: `store_id`

### Foreign Keys
- **film_actor**: `actor_id` (ref: actor), `film_id` (ref: film)
- **film_category**: `film_id` (ref: film), `category_id` (ref: category)
- **address**: `city_id` (ref: city)
- **city**: `country_id` (ref: country)
- **customer**: `store_id` (ref: store), `address_id` (ref: address)
- **inventory**: `film_id` (ref: film), `store_id` (ref: store)
- **rental**: `inventory_id` (ref: inventory), `customer_id` (ref: customer), `staff_id` (ref: staff)
- **payment**: `customer_id` (ref: customer), `staff_id` (ref: staff), `rental_id` (ref: rental)
- **store**: `manager_staff_id` (ref: staff), `address_id` (ref: address)

### Relationship Diagram
![DVD Rental ER Diagram](ERD.jpg)

## Part 2: JOIN Explanations and Solutions

**1. Display Customer Name, Email, City, and Country**
- *Solution*: Joined `customer` to `address` via `address_id`, then `address` to `city` via `city_id`, and finally `city` to `country` via `country_id`.
- *Why*: The customer table only has a link to their address. We have to traverse through the address to get the city, and then through the city to get the country name.

**2 & 3. Display every payment with Customer Name, Film Title, and Amount Paid**
- *Solution*: Joined `payment` to `customer` to get names. Then bridged to the film title by joining `payment` to `rental` (via `rental_id`), `rental` to `inventory` (via `inventory_id`), and finally `inventory` to `film` (via `film_id`).
- *Why*: There is no direct link between a payment and a film. A payment is made for a specific rental, that rental corresponds to a physical inventory item, and that item is a copy of a specific film.

**4. Find the Top 10 customers based on total amount spent**
- *Solution*: Joined `customer` to `payment` via `customer_id`. Grouped by customer and summed the `amount` column, then ordered descending and applied `LIMIT 10`.
- *Why*: We need the customer name from the `customer` table and the transaction amounts from the `payment` table. 

**5. Display each film with its Category and Rental Rate**
- *Solution*: Joined `film` to the bridge table `film_category` (via `film_id`), then to `category` (via `category_id`).
- *Why*: Films and categories have a many-to-many relationship bridged by `film_category`.

**6. Find all actors who appeared in each film**
- *Solution*: Joined `film` to the bridge table `film_actor` (via `film_id`), then to `actor` (via `actor_id`).
- *Why*: Similar to categories, actors and films have a many-to-many relationship requiring the bridge table.

**7. Count how many films belong to each category**
- *Solution*: Joined `category` to `film_category` (via `category_id`). Grouped by category name and counted `film_id`.
- *Why*: The bridge table holds the mapping of every film to a category.

**8. Which categories generated the highest revenue?**
- *Solution*: Chained joins from `category` -> `film_category` -> `inventory` -> `rental` -> `payment`. Grouped by category and summed `amount`.
- *Why*: Revenue is stored in `payment`. We have to link a payment back to the rental, to the inventory, to the film, and finally to its category.

**9. Find customers who have rented more than 20 films**
- *Solution*: Joined `customer` to `rental` (via `customer_id`). Grouped by customer, used `HAVING COUNT(rental_id) > 20`.
- *Why*: We need to count instances of rentals for each customer identifier.

**10. Which cities generated the highest rental revenue?**
- *Solution*: Joined `city` -> `address` -> `customer` -> `payment`. Grouped by city and summed `amount`.
- *Why*: We track revenue from the payment, linked to the customer making it, linked to their address, linked to the city.

### :star: Bonus Challenge
**Which actor has generated the highest total rental revenue?**
- *Shortest Path*: `actor` -> `film_actor` -> `inventory` -> `rental` -> `payment`.
- *Explanation*: We don't actually need the `film` table! The `film_actor` bridge gives us the `film_id` directly, which we can join straight to the `inventory` table. Then we follow inventory -> rental -> payment to get the revenue.

## Business Insights

1. **Category Popularity vs Revenue**: By analyzing the results of queries 7 and 8, you can identify if the categories with the most films are also the ones generating the most revenue. If a category has few films but high revenue, it represents an underserved market demand.
2. **Geographical Revenue Concentration**: The results of the city revenue query (#10) can help the business decide where to target marketing campaigns or where it might be most profitable to open a physical brick-and-mortar presence, based on where the highest paying customers live.
3. **High Value Customers (Whales)**: The results of query #4 and #9 identify the "whales" of the business. These top customers should be targeted with loyalty programs, special discounts, or early access to new releases to maintain their high retention and lifetime value.
