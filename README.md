# **CEN445 – Introduction to Data Visualization **

Asst. Prof. Dr. Mehmet SARIGÜL

# Finding Frequent Patterns Using FP-Growth Algorithm

## Team Members
- Tufan Akbaş - 2019555002
- Ömer Dinçer - 2020555402


## How To Use
- Clone this repository
- Go into the repository
- Install the necessary libraries
- Run "fp-growth_all.py" first



## Method
On that project, the FP-Growth algorithm is used to find frequent patterns in the 42.000 news dataset. The working principle is realized by creating the FP-Tree, a tree-based structure. In the first step, frequently occurring items in the dataset are identified. Then, the FP-Tree containing these items is created. The FP-Tree represents the frequent patterns in the dataset. When building the FP-Tree, the algorithm compresses the data in such a way that the frequent items are included, and thus efficiently analyzes the data set. Due to this simple structure and efficiency, FP-Growth is used for finding association rules in large data sets.

- **fp-growth_all**: It runs the algorithm on all news data and gives a single output.

- **fp-growth_categories**: It runs the algorithm on all news data, evaluates different categories within themselves and outputs the number of categories.


---


