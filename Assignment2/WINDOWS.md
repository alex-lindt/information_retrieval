# `trec_eval` for Windows:

1. Download the Cygwin setup file from https://cygwin.com/install.html
2. The basic installation does not automatically download all the required packages to install trec_eval. Double-click the the setup.exe file to start the installation of Cygwin. Follow the screen prompts up to the 'Cygwin Setup - Select Packages' screen. Select 'View -> Full', search for 'automake', go the line that shows package 'automake' and choose a version number under the tab 'New'. Before clicking on Next, do the same for 'gcc-core' and 'make'. Click Next and finish the installation.
3. Launch the Cygwin terminal. The default path at the start is '/home/<your_username>'.
4. Clone https://github.com/usnistgov/trec_eval.
5. Navigate to the install folder using 'cd trec_eval'.
6. Type 'make'. This should compile the code and create trec_eval.exe among other files.
7. Evaluating your results using trec_eval requires the results to be written to a file in a specific format. More info here: http://www.rafaelglater.com/en/post/learn-how-to-use-trec_eval-to-evaluate-your-information-retrieval-system
8. For example, in `tf-idf.py`, you could write the results in `overall_ser` to a file as:
```
# Write results to trec-style file
results_lines = []
for qid in overall_ser:
    for doc_id in overall_ser[qid]:
        results_lines.append(str(qid) + '\tQO\t' + doc_id + '\t0\t' + str(overall_ser[qid][doc_id]) + '\tSTANDARD\n')
with open('tf-idf_results.out', 'w') as f:
    f.writelines(results_lines)
```
9. You can obtain the metrics using
```
trec = TrecAPI('C:/cygwin64/home/<your_username>/trec_eval/trec_eval.exe',)
metrics = trec.evaluate(test_file_name='datasets/ap/qrels.tsv', prediction_file_name='tf-idf_results.out', metrics_to_capture={'map', 'ndcg'})
```
10. To execute your code, you will have to run it from the Cygwin terminal.
