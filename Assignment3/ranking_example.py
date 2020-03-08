import dataset
import ranking as rnk
import numpy as np

# we will rank 5 items with the scores:
scores = np.array([10., 8., 12., 9., 5.])

(ranking,
 inverted_ranking) = rnk.rank_and_invert(scores)

print('Ranking the scores: %s' % scores)
print('Resulting ranking: %s' % ranking)
print('This orders the scores as: %s' % scores[ranking])
print('an results in the inverted ranking: %s' % inverted_ranking)
print('The inverted ranking allows us to quickly see that:')
for i in range(scores.shape[0]):
  print('Item %d with score %0.02f has rank: %d' % (i, scores[i], inverted_ranking[i]))

print('It is also very useful for computing rank differences,')
print('for instance:')
for i, j in [(1,2), (2,4), (0,3)]:
  print('the difference between item %d (at rank %d)'
        ' and item %d (at rank %d) is %d' % (
          i, inverted_ranking[i],
          j, inverted_ranking[j],
          inverted_ranking[i]-inverted_ranking[j]))
