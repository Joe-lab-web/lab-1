import matplotlib.pyplot as plt
from easypyplot.barchart import draw
from easypyplot.pdf import plot_open

data = [
    [1, 1, 1, 1],
    [1.1, 1.2, 0.9, 0.8],
    [0.8, 0.9, 2.0, 1.1],
    [1.0, 1.1, 1.2, 1.2],
    [0.8, 0.9, 2.0, 1.1],
    [1.0, 1.1, 1.2, 1.2],
]

import numpy as np
num_tasks = 3

for i in range(num_tasks):
    with open('./tasks{}_run.log'.format(i+1), 'r') as f:
        blocks = []
        current_block = []
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                current_block.append(stripped_line)
            elif current_block:
                blocks.append(current_block)
                current_block = []
        if current_block:
            blocks.append(current_block)  
        # print(blocks)      
        
        for j in range(len(blocks)):
            for k,line in enumerate(blocks[j]):
                blocks[j][k] = line.split(' ')[-1]
        # print(blocks)
        
        blocks_array = np.array(blocks, dtype=int)
        blocks_array = blocks_array.transpose()
        # print(blocks_array)
        
        # caculating speedup
        def _calculate_speedup(original_time, new_time):
            return np.exp(np.log(original_time / new_time) / 2)

        # caculating speedup
        def calculate_speedup(original_time, new_time):
            return original_time / new_time
        
        original = blocks_array[0]
        optimized = blocks_array
        speedup_array = calculate_speedup(original, optimized)
        # print(speedup_array)
        
        data = speedup_array

        group_names = {1:["mnk", "mkn", "kmn", "nmk", "nkm", "knm"],
                       2:['mnk','t_mnk', 'mnkkmn_b32', 'mnk_lu2'],
                       3:['mnk','simd', 'o3', 'simd=o3']}
        
        with plot_open("lab_{}".format(i+1)) as fig:
            ax = fig.gca()
            draw(ax, data, breakdown=False,
                    # group_names=["mnk", "mkn", "kmn", "nmk", "nkm", "knm"],
                    # entry_names=["1.txt", "2.txt", "3.txt", "4.txt" ],
                    group_names = group_names.get(i+1),
                    entry_names = ["1.txt", "2.txt", "3.txt", "4.txt", '5.txt', '6.txt', '7.txt', '8.txt'],
                    )
            ax.set_xlabel('Different implementations of compute kernel')
            ax.set_ylabel('Speedup')
            ax.set_title('Task {}'.format(i+1))
