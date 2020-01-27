"""计时器"""


class Timer:
	"""Clock decorator"""

	def __init__(self, func):
		self.func = func

	def __call__(self, *args, **kwargs):
		from time import time
		t = time()
		self.func(*args, **kwargs)
		print("duration: {:.4f}".format(time() - t))
		return self
