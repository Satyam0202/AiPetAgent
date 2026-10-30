CREATE DATABASE IF NOT EXISTS company_db;
USE company_db;
CREATE TABLE IF NOT EXISTS employees (
    id INT NOT NULL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    department VARCHAR(30) NOT NULL,
    salary INT NOT NULL,
    age INT NOT NULL
);

INSERT IGNORE INTO employees (id, name, department, salary, age)
WITH RECURSIVE employee_numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1
    FROM employee_numbers
    WHERE n < 500
)
SELECT
    n,
    CONCAT('Employee ', LPAD(n, 3, '0')),
    CASE MOD(n - 1, 6)
        WHEN 0 THEN 'IT'
        WHEN 1 THEN 'HR'
        WHEN 2 THEN 'Finance'
        WHEN 3 THEN 'Sales'
        WHEN 4 THEN 'Marketing'
        ELSE 'Operations'
    END,
    30000 + MOD(n * 7919, 90000),
    21 + MOD(n * 7, 40)
FROM employee_numbers;
SELECT COUNT(*) AS employee_count FROM employees;
