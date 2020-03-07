# MVA_2020_NLP_TD2
Practical assignment of speech and natural language processing

Sentence tree parser by algorithm CYK

Before to run make sure that there are the following files in directory "data"

1) 'sequoia-corpus+fct.mrg_strict' - the parsed text which is used as train data
2) 'polyglot-fr.pkl' (downloadable here https://sites.google.com/site/rmyeid/projects/polyglot) - french vocabulary embeddings

To run use command ./run.sh --file <input file path(must be a .txt file with a sentence in each row)> --timeLimit <time limit for one sentece in seconds>
  
  
For evaluation the results use run the eval.py file
There are three arguments <\br>
--parsedOutput: Parsed text file path
--input: Input text file path
--output: The output file

