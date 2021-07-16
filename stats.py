import numpy as np


def mean(arr):
  """
    Calculates mean of given array/list
  """
  return sum(arr)/len(arr)


def median(arr):
  """
    Calculates median of given array/list
  """
  nums = sorted(arr)
  n, med = len(nums), None
  if n%2 == 0:
    med = (nums[n//2-1] + nums[n//2])/2
  else:
    med = nums[n//2]
  return med


def variance(arr):
  """
    Calculates variance of given array/list
  """
  nums = sorted(arr)
  mean_val = mean(arr)
  var = sum([(ni-mean_val)**2 for ni in nums])/len(nums)
  return var


def std_dev(arr):
  """
    Calculates std-dev of given array/list
  """
  var = variance(arr)
  return np.sqrt(var)


def iqr(arr):
  """
    Calculates the inter quartile range of given array/list
  """
  n = len(arr)
  arr = sorted(arr)
  med1 = median(arr[:n//2])
  med2 = median(arr[-1:-(n//2 +1):-1][::-1])
  return (med2 - med1)


def percentile(arr, p):
  """
    Calculates the given p percentile of given array/list
  """
  nums = sorted(arr)
  rank = round(len(nums)*p/100)
  return (nums[rank] + nums[rank-1])/2


def med_abs_deviation(arr):
  """
    Calculates the median abs deviation of given array/list
  """
  med = median(arr)
  meds = [abs(med - num) for num in arr]
  return median(meds)



if __name__ == '__main__':
  from scipy import stats

  nums = np.random.randint(100, size=(100))

  assert round(mean(nums),3) == round(np.mean(nums), 3), "Mean not correct"
  assert round(median(nums),3) == round(np.median(nums), 3), "Median not correct"
  assert round(variance(nums),3) == round(np.var(nums), 3), "Variance not correct"
  assert round(variance(nums)**0.5,3) == round(np.std(nums), 3), "Std-dev not correct"
  assert round(iqr(nums),3) == round(stats.iqr(nums, interpolation='midpoint'), 3), "IQR not correct"
  assert round(percentile(nums, 90),3) == round(np.percentile(nums, 90, interpolation='midpoint'), 3), "90 percentile not correct"
  assert round(percentile(nums, 99),3) == round(np.percentile(nums, 99, interpolation='midpoint'), 3), "99 percentile not correct"
  assert round(med_abs_deviation(nums),3) == round(stats.median_abs_deviation(nums), 3), "MAD not correct"

  print("Yeeey.. stats.py works.!")
