import os
import re
from pathlib import Path
from collections import defaultdict
import argparse
from tqdm import tqdm

def parse_eval_results(file_path):
    """Parse eval_results.txt and extract metrics"""
    metrics = {}
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # First, try to find "total best result" section (for clevr-change)
        total_best_section = re.search(r'------------total best result-------------\s*(.*?)$', content, re.DOTALL)
        
        if total_best_section:
            # Use total best result section if available (clevr-change format)
            metrics_text = total_best_section.group(1)
        else:
            # Try to find "semantic change best result" section (for clevr-dc format)
            semantic_best_section = re.search(r'------------semantic change best result-------------\s*(.*?)$', content, re.DOTALL)
            if semantic_best_section:
                metrics_text = semantic_best_section.group(1)
            else:
                # Fallback to original method (reading from top test results)
                test_section = re.search(r'test results.*?semantic change captions only(.*?)=========', content, re.DOTALL)
                if test_section:
                    metrics_text = test_section.group(1)
                else:
                    return metrics
        
        # Parse each metric
        metric_patterns = {
            'Bleu_1': r'Bleu_1: ([\d.]+)',
            'Bleu_2': r'Bleu_2: ([\d.]+)',
            'Bleu_3': r'Bleu_3: ([\d.]+)',
            'Bleu_4': r'Bleu_4: ([\d.]+)',
            'METEOR': r'METEOR: ([\d.]+)',
            'ROUGE_L': r'ROUGE_L: ([\d.]+)',
            'CIDEr': r'CIDEr: ([\d.]+)',
            'SPICE': r'SPICE: ([\d.]+)'
        }
        
        for metric_name, pattern in metric_patterns.items():
            match = re.search(pattern, metrics_text)
            if match:
                metrics[metric_name] = float(match.group(1)) * 100
                    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        
    return metrics

def find_test_output_dirs(root_path):
    """Find all test_output_* directories"""
    test_dirs = []
    for item in os.listdir(root_path):
        if item.startswith('test_output_') and os.path.isdir(os.path.join(root_path, item)):
            # Extract iteration number
            iter_match = re.match(r'test_output_(\d+)', item)
            if iter_match:
                iter_num = int(iter_match.group(1))
                test_dirs.append((iter_num, item))
    
    return sorted(test_dirs)  # Sort by iteration number

def analyze_experiment_results(exp_path):
    """Analyze all evaluation results in an experiment directory"""
    results = defaultdict(dict)
    
    # Find all test_output directories
    test_dirs = find_test_output_dirs(exp_path)
    
    if not test_dirs:
        print(f"No test_output_* directories found in {exp_path}")
        return None
    
    print(f"Found {len(test_dirs)} test output directories")
    
    # Process each test directory with progress bar
    for iter_num, dir_name in tqdm(test_dirs, desc="Processing iterations"):
        eval_file = os.path.join(exp_path, dir_name, 'captions', 'eval_results.txt')
        if os.path.exists(eval_file):
            metrics = parse_eval_results(eval_file)
            if metrics:
                for metric_name, value in metrics.items():
                    results[metric_name][iter_num] = value
        else:
            print(f"\nWarning: {eval_file} not found")
    
    return results

def get_metric_order(metric_name):
    """Define custom order for metrics: BLEU, METEOR, ROUGE, CIDEr, SPICE"""
    order_map = {
        'Bleu_1': 0,
        'Bleu_2': 1,
        'Bleu_3': 2,
        'Bleu_4': 3,
        'METEOR': 4,
        'ROUGE_L': 5,
        'CIDEr': 6,
        'SPICE': 7
    }
    return order_map.get(metric_name, 999)

def sort_metrics(metrics):
    """Sort metrics in desired order"""
    return sorted(metrics, key=get_metric_order)

def find_best_iterations(results):
    """Find best iteration for each metric"""
    best_iters = {}
    
    for metric_name, iter_values in results.items():
        if iter_values:
            # For all metrics, higher is better
            best_iter = max(iter_values.items(), key=lambda x: x[1])
            best_iters[metric_name] = {
                'iteration': best_iter[0],
                'value': best_iter[1]
            }
    
    return best_iters

def print_analysis_results(exp_path, results, best_iters):
    """Print formatted analysis results"""
    print(f"\n{'='*60}")
    print(f"Analysis Results for: {exp_path}")
    print(f"{'='*60}\n")
    
    # Print best iteration for each metric
    print("Best Iterations by Metric:")
    print("-" * 40)
    for metric_name in sort_metrics(results.keys()):
        if metric_name in best_iters:
            best_info = best_iters[metric_name]
            print(f"{metric_name:10s}: Iteration {best_info['iteration']:6d} (value: {best_info['value']:.1f})")
    
    # Collect unique best iterations
    unique_best_iters = set()
    for metric_info in best_iters.values():
        unique_best_iters.add(metric_info['iteration'])
    
    # Print detailed results for each best iteration
    print(f"\n\nDetailed Results for Best Iterations:")
    print("="*60)
    
    for iter_num in sorted(unique_best_iters):
        print(f"\nIteration {iter_num}:")
        print("-" * 30)
        
        # Check which metrics this iteration is best for
        best_for_metrics = []
        for metric_name, best_info in best_iters.items():
            if best_info['iteration'] == iter_num:
                best_for_metrics.append(metric_name)
        
        print(f"Best for: {', '.join(best_for_metrics)}")
        print("\nAll metrics:")
        
        # Print all metrics for this iteration
        for metric_name in sort_metrics(results.keys()):
            if iter_num in results[metric_name]:
                value = results[metric_name][iter_num]
                is_best = iter_num == best_iters.get(metric_name, {}).get('iteration')
                marker = " *BEST*" if is_best else ""
                print(f"  {metric_name:10s}: {value:.1f}{marker}")
    
    # Print summary statistics
    print(f"\n\nSummary Statistics:")
    print("="*40)
    print(f"Total iterations analyzed: {len(set().union(*[set(v.keys()) for v in results.values()]))}")
    print(f"Metrics tracked: {len(results)}")
    print(f"Unique best iterations: {len(unique_best_iters)}")

def analyze_multiple_experiments(exp_paths):
    """Analyze multiple experiment directories"""
    all_results = {}
    
    for exp_path in tqdm(exp_paths, desc="Analyzing experiments"):
        print(f"\n\nProcessing: {exp_path}")
        results = analyze_experiment_results(exp_path)
        if results:
            all_results[exp_path] = {
                'results': results,
                'best_iters': find_best_iterations(results)
            }
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description='Analyze evaluation results across iterations')
    parser.add_argument('exp_paths', type=str, nargs='+', help='Path(s) to experiment directory')
    parser.add_argument('--metric', type=str, default=None, 
                        help='Focus on specific metric (e.g., CIDEr, SPICE)')
    parser.add_argument('--all-progress', action='store_true', default=True,
                        help='Show progression for all metrics')
    parser.add_argument('--compare', action='store_true',
                        help='Compare results across multiple experiments')
    args = parser.parse_args()
    
    # Handle single or multiple experiments
    if len(args.exp_paths) == 1 and not args.compare:
        # Single experiment analysis
        results = analyze_experiment_results(args.exp_paths[0])
        
        if results:
            # Find best iterations
            best_iters = find_best_iterations(results)
            
            # Print analysis
            print_analysis_results(args.exp_paths[0], results, best_iters)
            
            # If specific metric requested, show its progression
            if args.metric and args.metric in results:
                print(f"\n\n{args.metric} Progression:")
                print("="*40)
                iter_values = results[args.metric]
                for iter_num in sorted(iter_values.keys()):
                    value = iter_values[iter_num]
                    is_best = iter_num == best_iters.get(args.metric, {}).get('iteration')
                    marker = " *BEST*" if is_best else ""
                    print(f"Iteration {iter_num:6d}: {value:.1f}{marker}")
            
            # If all progress requested, show all metrics
            if args.all_progress:
                print(f"\n\nAll Metrics Progression:")
                print("="*80)
                
                # Get all iterations
                all_iters = sorted(set().union(*[set(v.keys()) for v in results.values()]))
                
                # Print header
                header = "Iteration"
                for metric in sort_metrics(results.keys()):
                    header += f" | {metric:^8s}"
                print(header)
                print("-" * len(header))
                
                # Print values for each iteration
                for iter_num in all_iters:
                    row = f"{iter_num:9d}"
                    for metric in sort_metrics(results.keys()):
                        if iter_num in results[metric]:
                            value = results[metric][iter_num]
                            is_best = iter_num == best_iters.get(metric, {}).get('iteration')
                            if is_best:
                                row += f" | {value:7.1f}*"
                            else:
                                row += f" | {value:8.1f}"
                        else:
                            row += " |    -    "
                    print(row)
    else:
        # Multiple experiments analysis
        all_results = analyze_multiple_experiments(args.exp_paths)
        
        if all_results:
            # Print individual results
            for exp_path, data in all_results.items():
                print_analysis_results(exp_path, data['results'], data['best_iters'])
            
            # Compare across experiments if requested
            if args.compare and len(all_results) > 1:
                print("\n\n" + "="*80)
                print("COMPARISON ACROSS EXPERIMENTS")
                print("="*80)
                
                # Compare best values for each metric
                metrics = set()
                for data in all_results.values():
                    metrics.update(data['results'].keys())
                
                for metric in sort_metrics(metrics):
                    print(f"\n{metric}:")
                    print("-" * 40)
                    
                    exp_values = []
                    for exp_path, data in all_results.items():
                        if metric in data['best_iters']:
                            best_info = data['best_iters'][metric]
                            exp_name = os.path.basename(exp_path)
                            exp_values.append((exp_name, best_info['iteration'], best_info['value']))
                    
                    # Sort by value (descending)
                    exp_values.sort(key=lambda x: x[2], reverse=True)
                    
                    for i, (exp_name, iter_num, value) in enumerate(exp_values):
                        rank_marker = " [BEST]" if i == 0 else ""
                        print(f"  {exp_name:30s}: {value:.1f} (iter {iter_num}){rank_marker}")

if __name__ == "__main__":
    main()