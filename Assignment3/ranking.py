"""
File with helper-functions to generate rankings and associated rank functions.
"""
import numpy as np

def rank_and_invert(scores):
  """
  Generates a ranking and an inverted ranking,
  based on the given scores.

  Args:
      scores (array int/float): Scores for the items.

  Returns:
      array int: The ranking for the scores,
                 i.e. scores[ranking] sorts the scores
                 in descending order.
      array int: The inverted ranking,
                 this is simply the rank function,
                 i.e. for document i with score: scores[i]
                 its rank in the ranking is inverted[i],
                 (counting starts at 0).  
  """
  n_docs = scores.shape[0]
  noise = np.random.uniform(size=n_docs)
  # We use noise to shuffle items with equal scores.
  ranking = np.lexsort((noise, scores))[::-1]
  inverted = np.empty(n_docs, dtype=ranking.dtype)
  inverted[ranking] = np.arange(n_docs)
  return ranking, inverted

def data_split_rank_and_invert(scores, data_split):
  """
  Generates rankings and inverted rankings,
  based on the given scores for an entire dataset split.
  For instance, you can use this function to get
  all the rankings for the train, validation or test set.

  Args:
      scores (array int/float): Scores for all items in the split,
                                separeted using the query-ranges.
      data_split (DataFoldSplit): The data-split to generate rankings
                                  over, i.e. data.train, data.validation
                                  or data.test.
      
  Returns:
      array int: A concatenation of rankings for all queries,
                 in the split, i.e. to get the ranking for query qid:
                 s_i, e_i = data_split.query_range(qid)
                 query_ranking = ranking[s_i:e_i]
      array int: A concatenation of  inverted rankings for all queries,
                 (see the documentation for rank_and_invert),
                 to get the ranking for query qid:
                 s_i, e_i = data_split.query_range(qid)
                 query_ranking = ranking[s_i:e_i]  
  """
  ranking = np.zeros(data_split.num_docs(), dtype=np.int64)
  inverted = np.zeros(data_split.num_docs(), dtype=np.int64)
  for qid in np.arange(data_split.num_queries()):
    s_i, e_i = data_split.query_range(qid)
    q_scores = scores[s_i:e_i]
    (ranking[s_i:e_i],
     inverted[s_i:e_i]) = rank_and_invert(q_scores)
  return ranking, inverted