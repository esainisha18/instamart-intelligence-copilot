-- Q1. Completed orders above INR 500 grouped by customer, retaining totals above INR 8000.
SELECT
    customer_id,
    SUM(order_amount) AS total_qualifying_spend
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_qualifying_spend DESC, customer_id;

-- Q2. Completed-order revenue by customer city and product category.
SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS revenue_inr
FROM orders AS o
JOIN customers AS c ON c.customer_id = o.customer_id
JOIN products AS p ON p.product_id = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue_inr DESC, c.city, p.category;

-- Q3. Products priced above the average catalogue price.
SELECT
    product_id,
    product_name,
    category,
    price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC, product_id;

-- Q4. Customers with no completed orders.
SELECT
    c.customer_id,
    c.city
FROM customers AS c
LEFT JOIN orders AS o
    ON o.customer_id = c.customer_id
   AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Q5. Customers whose completed spend is above the average customer spend.
WITH customer_totals AS (
    SELECT
        customer_id,
        SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_spend AS (
    SELECT AVG(total_spend) AS average_total_spend
    FROM customer_totals
)
SELECT
    customer_totals.customer_id,
    customer_totals.total_spend,
    ROUND(average_spend.average_total_spend, 2) AS average_total_spend
FROM customer_totals
CROSS JOIN average_spend
WHERE customer_totals.total_spend > average_spend.average_total_spend
ORDER BY customer_totals.total_spend DESC, customer_totals.customer_id;

-- Q6. Completed spend ranked within each city using three ranking window functions.
WITH customer_totals AS (
    SELECT
        c.customer_id,
        c.city,
        SUM(o.order_amount) AS total_spend
    FROM orders AS o
    JOIN customers AS c ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT
    customer_id,
    city,
    total_spend,
    RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,
    DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_dense_rank,
    ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_row_number
FROM customer_totals
ORDER BY city, spend_rank, customer_id;

-- Q7. Daily completed revenue and its running cumulative total.
WITH daily_revenue AS (
    SELECT
        substr(order_ts, 1, 10) AS day,
        SUM(order_amount) AS daily_revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY substr(order_ts, 1, 10)
)
SELECT
    day,
    ROUND(daily_revenue, 2) AS daily_revenue_inr,
    ROUND(
        SUM(daily_revenue) OVER (ORDER BY day),
        2
    ) AS cumulative_revenue_inr
FROM daily_revenue
ORDER BY day;
