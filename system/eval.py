from PYEVALB import scorer, parser
import matplotlib.pyplot as plt
import numpy as np
import argparse

p = argparse.ArgumentParser( description='Argument parser' )

p.add_argument( '--parsedOutput', type=str, required=True, help='Parsed text file path' )
p.add_argument( '--output', type=str, required=True, help='The output file' )

args = p.parse_args()


parsed_output = open(args.parsedOutput, 'r').readlines()
test_output = open(args.output, 'r').readlines()


precisions = []
recalls = []
failures = 0

for i in range(len(test_output)):
    if test_output[i] == 'Cannot find valid parsing\n':
        failures += 1
    else:
        try:
            right_tree = parser.create_from_bracket_string(parsed_output[i][2:-2])
            output_tree = parser.create_from_bracket_string(test_output[i][2:-2])
            result = scorer.Scorer().score_trees(right_tree, output_tree)

            recalls.append(result.recall)
            precisions.append(result.prec)
        except:
            failures +=1

failure_pct = round(failures/len(test_output) * 100, 2)
print('Parsing failures for %s sentences (%s%%)' % (failures, failure_pct))
print('Average precision is: ' + str(np.mean(precisions)))
print('Average recall is: ' + str(np.mean(recalls)))
