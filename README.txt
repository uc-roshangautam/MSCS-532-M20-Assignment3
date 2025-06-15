Algorithm Analysis: Randomized Quicksort vs Hash Tables
Performance comparison of randomized vs deterministic quicksort and hash table implementation with chaining.
How to Run
python sort.py
Requirements:
•	Python 3.7+
•	NumPy: pip install numpy
What It Does
The program runs two main analyses:
1.	Quicksort Comparison - Tests randomized vs deterministic quicksort on different data types
2.	Hash Table Testing - Analyzes hash table performance with chaining collision resolution

Summary:
•	Randomized quicksort performs consistently well on all input types
•	Deterministic quicksort degrades severely on sorted/reverse data (up to 119x slower)
•	Random pivot selection prevents worst-case O(n²) behavior
Hash Table Results
•	Maintains O(1) performance with load factor management
•	Average chain length: 1.51 with maximum of 3
•	Dynamic resizing keeps load factor below 0.75 threshold
Output
The program displays:
•	Algorithm demonstrations on small arrays
•	Performance timing comparisons
•	Hash table statistics and collision analysis

