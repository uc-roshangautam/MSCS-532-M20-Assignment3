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
Key Findings
Quicksort Performance (Deterministic/Randomized Ratio)
Input Size	Random	Sorted	Reverse	Repeated
100	0.8x	2.4x	4.2x	0.7x
500	0.7x	8.3x	17.9x	1.0x
1000	0.5x	7.7x	28.1x	0.9x
2000	0.8x	29.3x	58.4x	0.9x
5000	0.7x	72.6x	119.4x	1.0x
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

