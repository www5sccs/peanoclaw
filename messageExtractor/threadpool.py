from Queue import Queue
from threading import Thread

class Worker(Thread):
    """Thread executing tasks from a given tasks queue"""
    def __init__(self, tasks):
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True
        self.start()
    
    def run(self):
        while True:
            task = self.tasks.get()
            try:
              task.run()
            except Exception, e:
              import traceback
              print "Fehler:" + str(e)
              traceback.print_exc()
              
            self.tasks.task_done()

class ThreadPool:
    """Pool of threads consuming tasks from a queue"""
    def __init__(self, num_threads):
        self.tasks = Queue(num_threads)
        for _ in range(num_threads): Worker(self.tasks)

    def add_task(self, task):
        """Add a task to the queue"""
        self.tasks.put(task)

    def wait_completion(self):
        """Wait for completion of all the tasks in the queue"""
        self.tasks.join()

