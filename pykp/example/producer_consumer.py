NUMBER_OF_PRODUCER_PROCESSES = 5
NUMBER_OF_CONSUMER_PROCESSES = 5

from multiprocessing import Process, Queue
import random, hashlib, time, os


class Consumer:
    def __init__(self):
        self.msg = None

    def consume_msg(self, queue):
        while True:
            print('Got into consumer method, with pid: %s' % os.getpid())
            # if queue.qsize() != 0:
            if queue.empty():
                self.msg = queue.get()
                print('got msg: %s' % self.msg)
            else:
                self.msg = None
                print('Queue looks empty')
            time.sleep(random.randrange(5, 10))


class Producer:
    def __init__(self):
        self.msg = None
        self.count = 0

    def produce_msg(self, queue):
        while True:
            self.count += 1
            print('Producing %d' % self.count)
            if self.count > 5:
                print("Producer terminated!")
                break

            print('Got into producer method, with pid: %s' % os.getpid())
            # self.msg = hashlib.md5(random.random().__str__()).hexdigest()
            self.msg = random.random().__str__()
            queue.put(self.msg)
            print('Produced msg: %s' % self.msg)
            time.sleep(random.randrange(5, 10))


if __name__ == "__main__":
    process_pool = []
    queue = Queue()

    producer = Producer()
    consumer = Consumer()

    for i in range(NUMBER_OF_PRODUCER_PROCESSES):
        print('Producer %d' % i)
        p = Process(target=producer.produce_msg, args=(queue,))
        process_pool.append(p)

    for i in range(NUMBER_OF_CONSUMER_PROCESSES):
        print('Consumer %d' % i)
        p = Process(target=consumer.consume_msg, args=(queue,))
        process_pool.append(p)

    for each in process_pool:
        each.start()