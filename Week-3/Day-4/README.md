# Music Store Business Intelligence Implementation

## Task 2: Segmentation Logic & Justification
Customers are categorized into four segments based on purchasing behavior:
- **Platinum**: Total Spend > $100 AND Unique Genres >= 5. *Justification*: These are high-value whales who also show broad taste, making them perfect targets for wide-scale catalog releases.
- **Gold**: Total Spend > $75 OR Total Invoices >= 10. *Justification*: Consistent buyers or high spenders. They keep the lights on and provide steady cash flow.
- **Silver**: Total Spend > $40. *Justification*: Average users who occasionally buy.
- **Bronze**: Everyone else. *Justification*: Low engagement or one-time purchasers.

## Task 3: Marketing Recommendation Strategy
Using the favorite genre calculated via `ROW_NUMBER()`, we dynamically assign campaigns:
- **Platinum**: Early access to new releases in their favorite genre (Rewards loyalty).
- **Gold**: Exclusive Album Bundles in their favorite genre (Encourages larger cart sizes).
- **Silver**: 15% Off all tracks in their favorite genre (Incentivizes moving up to Gold).
- **Bronze**: First purchase coupon for their favorite genre (Lowers the barrier to entry for their next purchase).

## Task 4: Country Ranking Methodology
The expansion score is a 100-point index calculated via Window Functions:
- **Total Revenue (40%)**: The primary indicator of a healthy market.
- **Average Revenue per Customer (30%)**: Indicates the purchasing power of the region.
- **Total Customers (30%)**: Indicates market penetration and scale.
*Methodology*: Each metric is normalized against the global maximum value using `MAX() OVER()` before weights are applied. The output of the pipeline highlights the top 3 countries as the safest bets for targeted physical/digital expansion.

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
