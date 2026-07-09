# Music Store Business Intelligence Implementation

## Task 1: Customer Spending Profiles
The foundation of the pipeline begins by building a reusable customer profile. To avoid Cartesian product duplication, the logic is strictly split into two pre-aggregation CTEs:
- **`Invoice_Agg`**: Calculates invoice-level metrics (`total_spent`, `total_invoices`, `avg_invoice_value`).
- **`Track_Agg`**: Calculates track-level metrics (`total_tracks_purchased`, `unique_genres`, `unique_artists`).
These are safely joined at the customer grain, ensuring all subsequent tasks have a clean, accurate dataset to pull from without recalculating totals.
## Task 2: Segmentation Logic & Justification
Customers are categorized into four segments based on purchasing behavior:
- **Platinum**: Total Spend > $41 AND Unique Artists >= 5. *Justification*: These are high-value whales who show broad tastes across many artists, making them perfect targets for wide-scale, premium catalog releases.
- **Gold**: Total Spend > $39 AND Unique Genres >= 4. *Justification*: Captures consistent buyers who explore a lot of different genres (genre diversity). They keep the lights on and provide steady cash flow.
- **Silver**: Total Spend > $37 OR Unique Artists >= 4. *Justification*: Average spenders or low-spenders who explore a lot of different artists (artist diversity). Their curiosity shows high potential for future targeted marketing to convert them to Gold.
- **Bronze**: Everyone else. *Justification*: Low engagement or one-time purchasers.

## Task 3: Marketing Recommendation Strategy
Using the favorite genre calculated via `ROW_NUMBER()`,  dynamically assign campaigns:
- **Platinum**: Early access to new releases in their favorite genre (Rewards loyalty).
- **Gold**: Exclusive Album Bundles in their favorite genre (Encourages larger cart sizes).
- **Silver**: 15% Off all tracks in their favorite genre (Incentivizes moving up to Gold).
- **Bronze**: First purchase coupon for their favorite genre (Lowers the barrier to entry for their next purchase).

## Task 4: Country Ranking Methodology
The expansion score is a 100-point index calculated via Window Functions using six critical metrics:
- **Total Revenue (30%)**: The primary indicator of market size and health.
- **Total Customers (20%)**: Indicates market penetration and scale.
- **Average Revenue per Customer (20%)**: Indicates the purchasing power of the average user in that region.
- **Average Invoice Value (10%)**: Shows how much users spend per individual transaction.
- **Number of Genres Purchased (10%)**: Measures the breadth of content consumption in the region.
- **Customer Diversity (10%)**: A count of distinct customer segments present, ensuring the market has a healthy mix of casual and VIP buyers.
*Methodology*: Each metric is normalized against the global maximum value using `MAX() OVER()` before weights are applied. The output of the pipeline highlights the top 3 countries as the safest bets for targeted physical/digital expansion.

## Task 5 & Bonus Challenge: Executive Dashboard Pipeline
To solve the final executive reporting requirement, the entire analysis has been refactored into a **single, chained data pipeline**. 
Instead of running isolated, disconnected queries, the script uses a continuous chain of 19 Common Table Expressions (CTEs) that dynamically build upon each other:
1. **Customer Profile** (`Invoice_Agg`, `Track_Agg`)
2. **Customer Segments**
3. **Favorite Genres** (`Genre_Counts`)
4. **Country Metrics & Ranking**
5. **Executive Aggregations** (`Employee_Revenue`, `Artist_Revenue`, `Album_Revenue`, `Segment_Agg`)

**Final Output**: The pipeline culminates in a single `UNION ALL` query that generates a unified Executive Dashboard containing all required metrics (Top Artist, Top Album, Segment Revenue, Country Contribution, etc.) in a standardized format without duplicating aggregate calculations.

## 5 Actionable Recommendations
1. **Focus retention efforts on Platinum and Gold members**, who likely drive the vast majority of total revenue (Pareto Principle).
2. **Execute hyper-targeted genre campaigns**. Since we now know every customer's favorite genre through our pipeline, generic marketing emails should be completely replaced with genre-specific recommendations.
3. **Expand localized digital presence in the Top 3 Ranked Countries**, as they show both high total revenue and high customer density based on our custom weighted score.
4. **Offer targeted discounts to Bronze members** using the "First purchase coupon" strategy specifically targeted at their identified favorite genre to convert them to active buyers.
5. **Analyze the top genres within the Platinum segment** to influence future licensing or artist signing decisions, maximizing ROI where the most money is spent.

## Challenges Faced
- **Challenge**: Avoiding duplicated aggregate values (Cartesian products) when joining `customer`, `invoice`, and `invoice_line` tables simultaneously. 
- **Solution**: Split the aggregations into two separate CTEs (`Invoice_Agg` and `Track_Agg`) and then joined them cleanly at the `customer_id` grain in `Customer_Profile`.
- **Challenge**: Normalizing scores in SQL without using nested subqueries in the `SELECT` clause, which can be inefficient and hard to read.
- **Solution**: Leveraged Window Functions (`MAX() OVER()`) to dynamically find the max value across the dataset and calculate relative percentages on the fly for the Country Ranking.
