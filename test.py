import sys
from predict import pred
if __name__ == "__main__":
	arg = str(sys.argv[1])
	arr = pred(arg)
	print(arr[0][0])