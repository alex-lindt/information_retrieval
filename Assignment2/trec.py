import os
import json
import subprocess
import collections


class TrecAPI:
    """
    API for TRECEval
    """

    def __init__(self, trec_path):
        """
        Create an instance for the TREC api
        :param trec_path: The path to the trec binary
        """
        assert os.path.exists(
            trec_path), "TREC binary doesn't exist at specified path"
        self.trec_path = os.path.abspath(trec_path)

    def evaluate(self, test_file_name, prediction_file_name, metrics_to_capture=None, granular=True):
        """
        Evaluate the given file against the test file
        :param test_file_name: The test file to evaluate against
        :param prediction_file_name: The file to evaluate
        :param metrics_to_capture: Which metrics to compute. If `None`, `ndcg_cut_10`, `map_cut_1000`, `P_5` and `recall_1000` are computed
        :param granular: If True, metrics are computed (returned) for all queries, otherwise overall performance is computed
        """
        # defaults
        if metrics_to_capture is None:
            metrics_to_capture = {"ndcg_cut_10",
                                  "map_cut_1000", "P_5", "recall_1000"}
        # put this into a try catch block since trec can fail
        try:
            command = [self.trec_path, "-m", "all_trec",
                       "-q", test_file_name, prediction_file_name]
            output = subprocess.check_output(command, universal_newlines=True)
            data = collections.defaultdict(dict)
            for line in output.split("\n"):
                # ignore empty lines
                if line.strip() == "":
                    continue
                metric, query, value = line.split("\t")
                if not granular and query != "all":
                    continue
                metric = metric.strip()
                # ignore metrics we don't care about
                if metric not in metrics_to_capture:
                    continue
                # relstring is a binary string, don't convert to float
                if metric not in {"relstring"}:
                    value = float(value)
                data[query][metric] = value
            if not granular:
                return data["all"]
            return data
        except subprocess.CalledProcessError as e:
            # just print out the error if something doesn't work
            print(e.output)
            return None
