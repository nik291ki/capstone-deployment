import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations, groupby
from mlxtend.frequent_patterns import apriori, association_rules as mlxtend_association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

class AssociationRulesPipeline:
    """
    A flexible pipeline for generating association rules using either Apriori (small datasets)
    or custom efficient method (large datasets).
    """
    
    def __init__(self, data_size_threshold=50000):
        """
        Initialize the pipeline.
        
        Parameters:
        -----------
        data_size_threshold : int
            Threshold for switching between Apriori and custom method
        """
        self.data_size_threshold = data_size_threshold
        self.rules = None
        self.frequent_itemsets = None
        self.method_used = None
        
    def _freq(self, itemset):
        """Frequency count for itemset"""
        if type(itemset) == pd.core.series.Series:
            return itemset.value_counts().rename("freq")
        else:
            return pd.Series(Counter(itemset)).rename("freq")
    
    def _order_count(self, order_item):
        """Number of unique orders"""
        return len(set(order_item.index))
    
    def _get_item_pairs(self, order_item):
        """Generator for item pairs one at a time"""
        order_item = order_item.reset_index().to_numpy()
        
        for order_id, order_object in groupby(order_item, lambda x: x[0]):
            item_list = [item[1] for item in order_object]
            
            for item_pair in combinations(item_list, 2):
                yield item_pair
    
    def _merge_item_stats(self, item_pairs, item_stats):
        """Frequency and support calculation"""
        return (item_pairs
                .merge(item_stats.rename(columns={"freq": "freqA", "support": "supportA"}), 
                       left_on="item_A", right_index=True)
                .merge(item_stats.rename(columns={"freq": "freqB", "support": "supportB"}), 
                       left_on="item_B", right_index=True))
    
    def _merge_item_name(self, rules, item_name):
        """Get name associated with item"""
        columns = ["itemA", "itemB", "freqAB", "supportAB", "freqA", "supportA", 
                  "freqB", "supportB", "confidenceAtoB", "confidenceBtoA", "lift"]
        
        rules = (rules
                .merge(item_name.rename(columns={"item_name":"itemA"}), 
                       left_on="item_A", right_on="item_id")
                .merge(item_name.rename(columns={"item_name":"itemB"}), 
                       left_on="item_B", right_on="item_id"))
        
        return rules[columns]
    
    def _custom_association_rules(self, order_item, min_support, verbose=True):
        """Custom association rules method for large datasets"""
        
        if verbose:
            print("Using Custom Method for Large Dataset")
            print("Starting order_item: {:22d}".format(len(order_item)))
        
        # Calculate item frequency and support
        item_stats = self._freq(order_item).to_frame("freq")
        item_stats['support'] = item_stats['freq'] / self._order_count(order_item) * 100
        
        # Filter from order_item items below min support
        qualifying_items = item_stats[item_stats['support'] >= min_support].index
        order_item = order_item[order_item.isin(qualifying_items)]
        
        if verbose:
            print("Items with support >= {}: {:15d}".format(min_support, len(qualifying_items)))
            print("Remaining order_item: {:21d}".format(len(order_item)))
        
        # Filter from order_item orders with less than 2 items
        order_size = self._freq(order_item.index)
        qualifying_orders = order_size[order_size >= 2].index
        order_item = order_item[order_item.index.isin(qualifying_orders)]
        
        if verbose:
            print("Remaining orders with 2+ items: {:11d}".format(len(qualifying_orders)))
            print("Remaining order_item: {:21d}".format(len(order_item)))
        
        # Recalculate item frequency and support
        item_stats = self._freq(order_item).to_frame("freq")
        item_stats['support'] = item_stats['freq'] / self._order_count(order_item) * 100
        
        # Get item pairs generator
        item_pair_gen = self._get_item_pairs(order_item)
        
        # Calculate item pair frequency and support
        item_pairs = self._freq(item_pair_gen).to_frame("freqAB")
        item_pairs['supportAB'] = item_pairs['freqAB'] / len(qualifying_orders) * 100
        
        if verbose:
            print("Item pairs: {:31d}".format(len(item_pairs)))
        
        # Filter from item_pairs those below min support
        item_pairs = item_pairs[item_pairs['supportAB'] >= min_support]
        
        if verbose:
            print("Item pairs with support >= {}: {:10d}\n".format(min_support, len(item_pairs)))
        
        # Create table of association rules and compute relevant metrics
        item_pairs = item_pairs.reset_index().rename(columns={'level_0': 'item_A', 'level_1': 'item_B'})
        item_pairs = self._merge_item_stats(item_pairs, item_stats)
        
        item_pairs['confidenceAtoB'] = item_pairs['supportAB'] / item_pairs['supportA']
        item_pairs['confidenceBtoA'] = item_pairs['supportAB'] / item_pairs['supportB']
        item_pairs['lift'] = item_pairs['supportAB'] / (item_pairs['supportA'] * item_pairs['supportB'])
        
        # Return association rules sorted by lift in descending order
        return item_pairs.sort_values('lift', ascending=False)
    
    def _apriori_method(self, transactions_list, min_support, min_confidence=0.5, min_lift=1.0, verbose=True):
        """Apriori method for small datasets"""
        
        if verbose:
            print("Using Apriori Method for Small Dataset")
            print(f"Processing {len(transactions_list)} transactions")
        
        # Convert to binary matrix
        te = TransactionEncoder()
        te_ary = te.fit(transactions_list).transform(transactions_list)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Find frequent itemsets
        frequent_itemsets = apriori(df, min_support=min_support/100, use_colnames=True, verbose=verbose)
        
        if len(frequent_itemsets) == 0:
            print("No frequent itemsets found. Try lowering min_support.")
            return pd.DataFrame()
        
        # Generate association rules
        rules = mlxtend_association_rules(frequent_itemsets, 
                                        metric="confidence", 
                                        min_threshold=min_confidence,
                                        num_itemsets=len(frequent_itemsets))
        
        # Filter by lift
        rules = rules[rules['lift'] >= min_lift]
        
        # Convert to consistent format
        rules_formatted = pd.DataFrame({
            'item_A': [list(antecedent)[0] if len(antecedent) == 1 else str(list(antecedent)) for antecedent in rules['antecedents']],
            'item_B': [list(consequent)[0] if len(consequent) == 1 else str(list(consequent)) for consequent in rules['consequents']],
            'supportAB': rules['support'] * 100,
            'supportA': rules['antecedent support'] * 100,
            'supportB': rules['consequent support'] * 100,
            'confidenceAtoB': rules['confidence'],
            'lift': rules['lift']
        })
        
        self.frequent_itemsets = frequent_itemsets
        
        if verbose:
            print(f"Generated {len(rules_formatted)} rules")
        
        return rules_formatted.sort_values('lift', ascending=False)
    
    def prepare_data(self, data, order_col='order_id', item_col='product_name', segment_filter=None):
        """
        Prepare data for association rules mining.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Transaction data
        order_col : str
            Column name for order/transaction ID
        item_col : str
            Column name for item/product
        segment_filter : dict, optional
            Filter for specific customer segment {'column': 'segment', 'value': 1}
            
        Returns:
        --------
        Prepared data in appropriate format
        """
        # Apply segment filter if provided
        if segment_filter:
            data = data[data[segment_filter['column']] == segment_filter['value']].copy()
            print(f"Filtered to segment {segment_filter['value']}: {len(data)} transactions")
        
        # Remove missing values
        data = data.dropna(subset=[order_col, item_col])
        
        # Determine data size and format
        data_size = len(data)
        print(f"Data size: {data_size} transactions")
        
        if data_size <= self.data_size_threshold:
            # Small dataset - prepare for Apriori
            transactions_list = data.groupby(order_col)[item_col].apply(list).tolist()
            return 'apriori', transactions_list
        else:
            # Large dataset - prepare for custom method
            order_item = data.set_index(order_col)[item_col]
            return 'custom', order_item
    
    def generate_rules(self, data, order_col='order_id', item_col='product_name', 
                      min_support=1.0, min_confidence=0.5, min_lift=1.0, 
                      segment_filter=None, verbose=True):
        """
        Generate association rules using the most appropriate method.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Transaction data
        order_col : str
            Column name for order/transaction ID
        item_col : str
            Column name for item/product
        min_support : float
            Minimum support threshold (percentage)
        min_confidence : float
            Minimum confidence threshold
        min_lift : float
            Minimum lift threshold
        segment_filter : dict, optional
            Filter for specific customer segment
        verbose : bool
            Whether to print progress information
            
        Returns:
        --------
        pd.DataFrame : Association rules
        """
        
        # Prepare data
        method, prepared_data = self.prepare_data(data, order_col, item_col, segment_filter)
        self.method_used = method
        
        # Generate rules using appropriate method
        if method == 'apriori':
            self.rules = self._apriori_method(prepared_data, min_support, min_confidence, min_lift, verbose)
        else:
            self.rules = self._custom_association_rules(prepared_data, min_support, verbose)
            # Apply additional filters for custom method
            if min_confidence > 0:
                self.rules = self.rules[self.rules['confidenceAtoB'] >= min_confidence]
            if min_lift > 1.0:
                self.rules = self.rules[self.rules['lift'] >= min_lift]
        
        return self.rules
    
    def get_top_rules(self, n=10, metric='lift'):
        """
        Get top N rules by specified metric.
        
        Parameters:
        -----------
        n : int
            Number of top rules to return
        metric : str
            Metric to sort by ('lift', 'confidenceAtoB', 'supportAB')
            
        Returns:
        --------
        pd.DataFrame : Top rules
        """
        if self.rules is None or len(self.rules) == 0:
            return pd.DataFrame()
        
        return self.rules.nlargest(n, metric)
    
    def get_rules_for_item(self, item_name, as_antecedent=True, as_consequent=True):
        """
        Get rules involving a specific item.
        
        Parameters:
        -----------
        item_name : str
            Name of the item to search for
        as_antecedent : bool
            Include rules where item is antecedent
        as_consequent : bool
            Include rules where item is consequent
            
        Returns:
        --------
        pd.DataFrame : Filtered rules
        """
        if self.rules is None or len(self.rules) == 0:
            return pd.DataFrame()
        
        conditions = []
        
        if as_antecedent:
            conditions.append(self.rules['item_A'].str.contains(item_name, case=False, na=False))
        
        if as_consequent:
            conditions.append(self.rules['item_B'].str.contains(item_name, case=False, na=False))
        
        if conditions:
            combined_condition = conditions[0]
            for condition in conditions[1:]:
                combined_condition = combined_condition | condition
            
            return self.rules[combined_condition].sort_values('lift', ascending=False)
        
        return pd.DataFrame()
    
    def export_rules(self, filepath, format='csv'):
        """
        Export rules to file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the file
        format : str
            Export format ('csv', 'json', 'excel')
        """
        if self.rules is None or len(self.rules) == 0:
            print("No rules to export.")
            return
        
        if format == 'csv':
            self.rules.to_csv(filepath, index=False)
        elif format == 'json':
            self.rules.to_json(filepath, orient='records', indent=2)
        elif format == 'excel':
            self.rules.to_excel(filepath, index=False)
        else:
            raise ValueError("Unsupported format. Use 'csv', 'json', or 'excel'.")
        
        print(f"Rules exported to {filepath}")
    
    def summary_stats(self):
        """
        Get summary statistics of the generated rules.
        
        Returns:
        --------
        dict : Summary statistics
        """
        if self.rules is None or len(self.rules) == 0:
            return {"message": "No rules generated"}
        
        stats = {
            "method_used": self.method_used,
            "total_rules": len(self.rules),
            "avg_support": self.rules['supportAB'].mean(),
            "avg_confidence": self.rules['confidenceAtoB'].mean(),
            "avg_lift": self.rules['lift'].mean(),
            "max_lift": self.rules['lift'].max(),
            "min_lift": self.rules['lift'].min(),
        }
        
        return stats


# Example usage and testing function
def test_pipeline():
    """Test function to demonstrate pipeline usage"""
    
    # Create sample data
    np.random.seed(42)
    
    # Small dataset example
    small_data = pd.DataFrame({
        'order_id': np.repeat(range(100), 3),
        'product_name': np.random.choice(['Apple', 'Banana', 'Orange', 'Milk', 'Bread', 'Eggs'], 300),
        'customer_segment': np.random.choice([1, 2, 3], 300)
    })
    
    # Initialize pipeline
    pipeline = AssociationRulesPipeline(data_size_threshold=200)
    
    print("=== Testing with Small Dataset (Apriori) ===")
    rules_small = pipeline.generate_rules(small_data, min_support=5.0, min_confidence=0.3)
    print(f"Generated {len(rules_small)} rules")
    print("\nTop 5 rules:")
    print(pipeline.get_top_rules(5))
    
    print("\n=== Testing with Segment Filter ===")
    segment_rules = pipeline.generate_rules(
        small_data, 
        min_support=3.0, 
        segment_filter={'column': 'customer_segment', 'value': 1}
    )
    print(f"Generated {len(segment_rules)} rules for segment 1")
    
    print("\n=== Summary Statistics ===")
    print(pipeline.summary_stats())

if __name__ == "__main__":
    test_pipeline()