from concurrent.futures import ProcessPoolExecutor
import time

def do_something(x):
    print(x)
    time.sleep(x)
    print("done: "+str(x))
    

with ProcessPoolExecutor(6) as executor:
    # these return immediately and are executed in parallel, on separate processes
    executor.submit(do_something, 1)
    executor.submit(do_something, 2)
    executor.submit(do_something, 3)
    executor.submit(do_something, 4)
    executor.submit(do_something, 5)
    executor.submit(do_something, 6)