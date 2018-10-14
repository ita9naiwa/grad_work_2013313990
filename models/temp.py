import queue

q = queue.Queue()

q.put(1)
q.put(2)

for e in q:
    print(e)
