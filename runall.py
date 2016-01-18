import subprocess as sp

print "Running with dropout:"
sp.call(['python', 'gpu_cpu_compare_with_dropout.py'])

print "\n\nRunning without dropout:"
sp.call(['python', 'gpu_cpu_compare_without_dropout.py'])
