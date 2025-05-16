# SecVulEval
SecVulEval is a dataset of C/C++ vulnerabilities.

## Run Vulnerability Detection
To run the vulnerability detection experiments, first install the following packages.
```
torch==2.5.1
transformers==4.47.0
accelerate==1.3.0   # for automatic GPU distribution if using multi-GPU
openai
anthropic
tree-sitter
tree-sitter-c==0.23.4
tree-sitter-cpp==0.23.4
openpyxl
```

Also, have the following variables in your environment.
```
export OPENAI_API_KEY=<your-api-key>
export ANTHROPIC_API_KEY=<your-api-key>
export HF_TOKEN=<your-access-token>
```
or have them in the following files as a JSON with `api_key` as key and the notebook cell will store it in the variable in runtime.-
```
creds/openai_api_key.json
creds/anthropic_api_key.json
creds/hf_access_token.json
```
Then open the `orchestrator.ipynb` notebook. Detailed instruction on how to run the cells and explanation is given in the notebook.

To replicate our results, we have added the `random_subset.json` file used in our experiment. If you want to use different random subsets, run the `random_subset.py` file first. The script will overwrite the `random_subset.json` file. So be careful if you want to keep both files. You can switch between any subsets by changing the `dataset` variable in the notebook.

Note: The first time you run the experiment the context agent may take much longer. This is because it has to extract symbols from their original project. After the first run it will be cached in `symcache.sqlite` so future runs will be faster.