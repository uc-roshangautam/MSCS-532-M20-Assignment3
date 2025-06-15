import random
import time
import sys
import numpy as np
from typing import List, Any, Optional

# Increase recursion limit for testing purposes
sys.setrecursionlimit(10000)

class RandomizedQuicksort:
    """
    Implementation of Randomized Quicksort algorithm with performance analysis
    """
    
    def __init__(self):
        self.comparisons = 0
        self.swaps = 0
    
    def reset_counters(self):
        """Reset performance counters"""
        self.comparisons = 0
        self.swaps = 0
    
    def randomized_quicksort(self, arr: List[int], low: int = 0, high: int = None) -> List[int]:
        """
        Randomized Quicksort implementation
        
        Args:
            arr: Array to sort
            low: Starting index
            high: Ending index
            
        Returns:
            Sorted array
        """
        if high is None:
            high = len(arr) - 1
            
        if low < high:
            # Randomly choose pivot and partition
            pi = self._randomized_partition(arr, low, high)
            
            # Recursively sort elements before and after partition
            self.randomized_quicksort(arr, low, pi - 1)
            self.randomized_quicksort(arr, pi + 1, high)
        
        return arr
    
    def deterministic_quicksort(self, arr: List[int], low: int = 0, high: int = None, depth: int = 0) -> List[int]:
        """
        Deterministic Quicksort (first element as pivot) for comparison
        Uses depth limiting to prevent stack overflow
        """
        if high is None:
            high = len(arr) - 1
            
        # Use iterative approach for large arrays or deep recursion
        if depth > 100 or high - low > 1000:
            return self._iterative_quicksort(arr, low, high)
            
        if low < high:
            # Use first element as pivot (swap to end first)
            arr[low], arr[high] = arr[high], arr[low]
            self.swaps += 1
            pi = self._partition(arr, low, high)
            
            self.deterministic_quicksort(arr, low, pi - 1, depth + 1)
            self.deterministic_quicksort(arr, pi + 1, high, depth + 1)
        
        return arr
    
    def _iterative_quicksort(self, arr: List[int], low: int, high: int) -> List[int]:
        """
        Iterative implementation of quicksort to avoid recursion depth issues
        """
        # Create an explicit stack
        stack = [(low, high)]
        
        while stack:
            low, high = stack.pop()
            
            if low < high:
                # Use first element as pivot (swap to end first)
                arr[low], arr[high] = arr[high], arr[low]
                self.swaps += 1
                pi = self._partition(arr, low, high)
                
                # Push subproblems onto stack
                stack.append((low, pi - 1))
                stack.append((pi + 1, high))
        
        return arr
    
    def _randomized_partition(self, arr: List[int], low: int, high: int) -> int:
        """
        Randomly select pivot and partition array
        """
        # Randomly choose pivot
        random_index = random.randint(low, high)
        
        # Swap random element with last element
        arr[random_index], arr[high] = arr[high], arr[random_index]
        self.swaps += 1
        
        return self._partition(arr, low, high)
    
    def _partition(self, arr: List[int], low: int, high: int) -> int:
        """
        Partition function for quicksort
        """
        # Choose last element as pivot
        pivot = arr[high]
        
        # Index of smaller element
        i = low - 1
        
        for j in range(low, high):
            self.comparisons += 1
            # If current element is smaller than or equal to pivot
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                self.swaps += 1
        
        # Place pivot in correct position
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        self.swaps += 1
        
        return i + 1


class HashTableWithChaining:
    """
    Hash Table implementation using chaining for collision resolution
    """
    
    def __init__(self, initial_capacity: int = 16):
        self.capacity = initial_capacity
        self.size = 0
        self.buckets = [[] for _ in range(self.capacity)]
        self.load_factor_threshold = 0.75
    
    def _hash_function(self, key: Any) -> int:
        """
        Universal hash function implementation
        Using polynomial rolling hash for strings and modular arithmetic
        """
        if isinstance(key, str):
            hash_value = 0
            a = 31  # Prime number for polynomial rolling hash
            for char in key:
                hash_value = (hash_value * a + ord(char)) % self.capacity
            return hash_value
        elif isinstance(key, int):
            # Simple hash for integers
            return key % self.capacity
        else:
            # Hash based on string representation
            return hash(str(key)) % self.capacity
    
    def insert(self, key: Any, value: Any) -> None:
        """
        Insert key-value pair into hash table
        
        Args:
            key: Key to insert
            value: Value associated with key
        """
        index = self._hash_function(key)
        bucket = self.buckets[index]
        
        # Check if key already exists
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)  # Update existing
                return
        
        # Add new key-value pair
        bucket.append((key, value))
        self.size += 1
        
        # Check if resize is needed
        if self.size > self.capacity * self.load_factor_threshold:
            self._resize()
    
    def search(self, key: Any) -> Optional[Any]:
        """
        Search for value associated with key
        
        Args:
            key: Key to search for
            
        Returns:
            Value if found, None otherwise
        """
        index = self._hash_function(key)
        bucket = self.buckets[index]
        
        for k, v in bucket:
            if k == key:
                return v
        
        return None
    
    def delete(self, key: Any) -> bool:
        """
        Delete key-value pair from hash table
        
        Args:
            key: Key to delete
            
        Returns:
            True if deleted, False if not found
        """
        index = self._hash_function(key)
        bucket = self.buckets[index]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                self.size -= 1
                return True
        
        return False
    
    def _resize(self) -> None:
        """
        Resize hash table when load factor exceeds threshold
        """
        old_buckets = self.buckets
        self.capacity *= 2
        self.size = 0
        self.buckets = [[] for _ in range(self.capacity)]
        
        # Rehash all elements
        for bucket in old_buckets:
            for key, value in bucket:
                self.insert(key, value)
    
    def get_load_factor(self) -> float:
        """Get current load factor"""
        return self.size / self.capacity
    
    def get_statistics(self) -> dict:
        """Get hash table statistics"""
        non_empty_buckets = sum(1 for bucket in self.buckets if bucket)
        max_chain_length = max(len(bucket) for bucket in self.buckets) if self.buckets else 0
        avg_chain_length = self.size / non_empty_buckets if non_empty_buckets > 0 else 0
        
        return {
            'size': self.size,
            'capacity': self.capacity,
            'load_factor': self.get_load_factor(),
            'non_empty_buckets': non_empty_buckets,
            'max_chain_length': max_chain_length,
            'avg_chain_length': avg_chain_length
        }


class PerformanceAnalyzer:
    """
    Class for analyzing and comparing algorithm performance
    """
    
    @staticmethod
    def generate_test_data(size: int, data_type: str) -> List[int]:
        """
        Generate test data of different types
        
        Args:
            size: Size of array
            data_type: Type of data ('random', 'sorted', 'reverse', 'repeated')
            
        Returns:
            Generated array
        """
        if data_type == 'random':
            return [random.randint(1, size * 10) for _ in range(size)]
        elif data_type == 'sorted':
            return list(range(1, size + 1))
        elif data_type == 'reverse':
            return list(range(size, 0, -1))
        elif data_type == 'repeated':
            return [random.randint(1, 10) for _ in range(size)]
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    @staticmethod
    def time_sorting_algorithm(sort_func, arr: List[int]) -> tuple:
        """
        Time a sorting algorithm and return execution time and performance metrics
        """
        arr_copy = arr.copy()
        
        start_time = time.perf_counter()
        sort_func(arr_copy)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        return execution_time, arr_copy
    
    @staticmethod
    def time_sorting_algorithm_with_timeout(sort_func, arr: List[int], timeout: float = 10.0) -> tuple:
        """
        Time a sorting algorithm with timeout protection
        Returns (execution_time, sorted_array) or (inf, None) if timeout
        """
        import threading
        import queue
        
        arr_copy = arr.copy()
        result_queue = queue.Queue()
        
        def run_sort():
            try:
                start_time = time.perf_counter()
                sort_func(arr_copy)
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                result_queue.put((execution_time, arr_copy))
            except Exception as e:
                result_queue.put((float('inf'), None))
        
        # Start sorting in a separate thread
        sort_thread = threading.Thread(target=run_sort)
        sort_thread.daemon = True
        sort_thread.start()
        
        # Wait for result with timeout
        sort_thread.join(timeout)
        
        if sort_thread.is_alive():
            # Timeout occurred
            return float('inf'), None
        else:
            # Get result from queue
            try:
                return result_queue.get_nowait()
            except queue.Empty:
                return float('inf'), None
    
    @staticmethod
    def compare_quicksort_variants():
        """
        Compare randomized vs deterministic quicksort performance
        Tests both algorithms on all four data types
        """
        sizes = [100, 500, 1000, 2000, 5000]
        data_types = ['random', 'sorted', 'reverse', 'repeated']
        
        results = {}
        
        for data_type in data_types:
            results[data_type] = {'sizes': [], 'randomized': [], 'deterministic': []}
            
            for size in sizes:
                print(f"Testing {data_type} data, size {size}...")
                
                # Generate test data
                test_data = PerformanceAnalyzer.generate_test_data(size, data_type)
                
                # Test randomized quicksort
                rq = RandomizedQuicksort()
                rq.reset_counters()
                rand_time, _ = PerformanceAnalyzer.time_sorting_algorithm_with_timeout(
                    rq.randomized_quicksort, test_data, timeout=10.0
                )
                
                # Test deterministic quicksort  
                dq = RandomizedQuicksort()
                dq.reset_counters()
                det_time, _ = PerformanceAnalyzer.time_sorting_algorithm_with_timeout(
                    dq.deterministic_quicksort, test_data, timeout=10.0
                )
                
                results[data_type]['sizes'].append(size)
                results[data_type]['randomized'].append(rand_time)
                results[data_type]['deterministic'].append(det_time)
                
                # Show immediate results
                rand_str = f"{rand_time:.6f}s" if rand_time != float('inf') else "TIMEOUT"
                det_str = f"{det_time:.6f}s" if det_time != float('inf') else "TIMEOUT"
                print(f"  -> Randomized: {rand_str}, Deterministic: {det_str}")
                
                if rand_time != float('inf') and det_time != float('inf') and rand_time > 0:
                    ratio = det_time / rand_time
                    print(f"  -> Deterministic is {ratio:.1f}x slower")
        
        return results
    
    @staticmethod
    def test_hash_table_performance():
        """
        Test hash table performance with different load factors
        """
        hash_table = HashTableWithChaining()
        
        # Test insertion performance
        insert_times = []
        search_times = []
        load_factors = []
        
        test_keys = [f"key_{i}" for i in range(1000)]
        test_values = [f"value_{i}" for i in range(1000)]
        
        for i in range(0, 1000, 50):
            # Insert batch of items
            start_time = time.perf_counter()
            for j in range(i, min(i + 50, 1000)):
                hash_table.insert(test_keys[j], test_values[j])
            insert_time = time.perf_counter() - start_time
            
            # Test search performance
            search_keys = random.sample(test_keys[:i+50], min(50, i+50))
            start_time = time.perf_counter()
            for key in search_keys:
                hash_table.search(key)
            search_time = time.perf_counter() - start_time
            
            insert_times.append(insert_time)
            search_times.append(search_time)
            load_factors.append(hash_table.get_load_factor())
        
        return {
            'load_factors': load_factors,
            'insert_times': insert_times,
            'search_times': search_times,
            'final_stats': hash_table.get_statistics()
        }


def main():
    """
    Main function to run the analysis
    """
    print("Algorithm Efficiency and Scalability Analysis")
    print("=" * 50)
    
    # Part 1: Randomized Quicksort Analysis
    print("\nPart 1: Randomized Quicksort Analysis")
    print("-" * 40)
    
    # Test basic functionality
    test_array = [64, 34, 25, 12, 22, 11, 90]
    print(f"Original array: {test_array}")
    
    rqs = RandomizedQuicksort()
    sorted_array = rqs.randomized_quicksort(test_array.copy())
    print(f"Sorted array: {sorted_array}")
    print(f"Comparisons: {rqs.comparisons}, Swaps: {rqs.swaps}")
    
    # Verify deterministic quicksort works on small arrays
    print(f"\nTesting deterministic quicksort:")
    dqs = RandomizedQuicksort()
    det_sorted = dqs.deterministic_quicksort(test_array.copy())
    print(f"Deterministic sorted: {det_sorted}")
    print(f"Comparisons: {dqs.comparisons}, Swaps: {dqs.swaps}")
    
    # Demonstrate the difference with sorted input
    sorted_test = list(range(1, 11))  # [1, 2, 3, ..., 10]
    print(f"\nTesting with pre-sorted array: {sorted_test}")
    
    # Randomized on sorted data
    rqs_sorted = RandomizedQuicksort()
    rqs_sorted.randomized_quicksort(sorted_test.copy())
    print(f"Randomized on sorted - Comparisons: {rqs_sorted.comparisons}")
    
    # Deterministic on sorted data
    dqs_sorted = RandomizedQuicksort()
    dqs_sorted.deterministic_quicksort(sorted_test.copy())
    print(f"Deterministic on sorted - Comparisons: {dqs_sorted.comparisons}")
    print(f"Ratio: {dqs_sorted.comparisons / rqs_sorted.comparisons:.1f}x more comparisons")
    
    # Performance comparison
    print("\nPerformance Comparison (Randomized vs Deterministic Quicksort):")
    print("Testing all data types for both algorithms...")
    results = PerformanceAnalyzer.compare_quicksort_variants()
    
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    for data_type, data in results.items():
        print(f"\n{data_type.upper()} DATA:")
        print("-" * 40)
        for i, size in enumerate(data['sizes']):
            rand_time = data['randomized'][i]
            det_time = data['deterministic'][i]
            
            rand_str = f"{rand_time:.6f}s" if rand_time != float('inf') else "TIMEOUT"
            det_str = f"{det_time:.6f}s" if det_time != float('inf') else "TIMEOUT"
            
            print(f"Size {size:4d}: Randomized={rand_str:>12}, Deterministic={det_str:>12}")
    
    print("\n" + "="*60)
    
    # Part 2: Hash Table with Chaining
    print("\n\nPart 2: Hash Table with Chaining Analysis")
    print("-" * 45)
    
    # Test basic functionality
    ht = HashTableWithChaining()
    
    # Insert test data
    test_data = [("apple", 1), ("banana", 2), ("cherry", 3), ("date", 4)]
    print("Inserting test data...")
    for key, value in test_data:
        ht.insert(key, value)
        print(f"Inserted ({key}, {value})")
    
    # Test search
    print("\nSearch operations:")
    for key, _ in test_data:
        result = ht.search(key)
        print(f"Search '{key}': {result}")
    
    print(f"Search 'grape': {ht.search('grape')}")
    
    # Display statistics
    stats = ht.get_statistics()
    print(f"\nHash Table Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Performance analysis
    print("\nHash Table Performance Analysis:")
    perf_results = PerformanceAnalyzer.test_hash_table_performance()
    
    print(f"Final hash table statistics:")
    for key, value in perf_results['final_stats'].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()