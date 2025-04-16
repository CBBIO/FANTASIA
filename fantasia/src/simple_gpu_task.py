import multiprocessing
import pickle
import threading
from abc import abstractmethod

import pika
from pika import PlainCredentials
from protein_metamorphisms_is.tasks.queue import QueueTaskInitializer


class SimpleGPUTaskInitializer(QueueTaskInitializer):

    def __init__(self, conf, session_required=True):
        """
        Initialize the GPUTaskInitializer.

        This constructor initializes the configuration, and if required, sets up
        a database session. It also prepares the RabbitMQ connection parameters and
        initializes the stop event for managing worker processes.

        Args:
            conf (dict): Configuration dictionary.
            session_required (bool): Whether a database session is required.
                                     If True, the session is initialized.
        """
        super().__init__(conf, session_required)
        self.stop_event = multiprocessing.Event()

    def setup_rabbitmq(self):
        try:
            credentials = PlainCredentials(self.conf['rabbitmq_user'], self.conf['rabbitmq_password'])
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=self.conf['rabbitmq_host'], credentials=credentials))
            self.channel = self.connection.channel()

            # 👇 Esta es la única cola que vas a usar para procesamiento
            self.channel.queue_declare(queue=self.computing_queue, durable=True)
            self.channel.queue_declare(queue=self.inserting_queue, durable=True)

        except Exception:
            raise

    def start_workers(self):
        """
        Start the worker processes, but run the processor directly on the main thread
        to avoid CUDA reinitialization issues.
        """
        try:
            self.setup_rabbitmq()

            # Iniciamos hilo para monitorización
            monitor_thread = threading.Thread(target=self.monitor_queues)
            monitor_thread.start()
            self.threads.append(monitor_thread)

            # Lanza el inserter como proceso aparte
            db_inserter_process = multiprocessing.Process(target=self.run_db_inserter_worker, args=(self.stop_event,))
            db_inserter_process.start()
            self.processes.append(db_inserter_process)

            # 👇 Ejecuta el procesamiento directamente (evita fork)
            self.logger.info("Running processor in main process to avoid CUDA errors.")
            self.run_processor_worker(self.stop_event)

            db_inserter_process.join()

        finally:
            self.cleanup()

    def cleanup(self):
        """
        Clean up resources and stop worker processes.

        This method ensures that all worker processes and threads are properly terminated.
        """
        self.stop_event.set()
        for t in self.threads:
            t.join()
        for p in self.processes:
            p.terminate()  # Ensure all processes are terminated properly

    def run_processor_worker(self, stop_event):
        """
        Run the processor worker: consumes messages and processes tasks using GPU.

        This version no gestiona modelos, simplemente ejecuta `callback` para cada tarea.
        """
        self.logger.info("Processor worker (GPU) started.")
        with self._create_rabbitmq_connection() as channel:
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(
                queue=self.computing_queue,
                on_message_callback=self.callback
            )
            self.consume_messages(channel, stop_event)
        self.logger.info("Processor worker (GPU) stopped.")

    def publish_task(self, batch_data):
        """
        Publish a task to the GPU processing queue.

        This method serializes the task data and publishes it to the appropriate
        queue for the specified model type.

        Args:
            batch_data (any): The task data to be processed.
            model_type (str): The type of model for which the task is intended.
        """
        if not isinstance(batch_data, bytes):
            batch_data = pickle.dumps(batch_data)
        if not self.channel or not self.channel.is_open:
            self.setup_rabbitmq()
        queue_name = f"{self.computing_queue}"
        self.channel.basic_publish(exchange='', routing_key=queue_name, body=batch_data)

    def _create_rabbitmq_connection(self):
        """
        Create a connection to RabbitMQ for GPU tasks.

        This method establishes a connection to RabbitMQ using the provided
        connection parameters.

        Returns:
            pika.channel.Channel: The RabbitMQ channel for processing tasks.
        """
        credentials = PlainCredentials(self.conf['rabbitmq_user'], self.conf['rabbitmq_password'])
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=self.conf['rabbitmq_host'], credentials=credentials))
        channel = connection.channel()
        return channel

    # Abstract methods inherited from QueueTaskInitializer
    @abstractmethod
    def enqueue(self):
        pass

    @abstractmethod
    def process(self, target):
        pass

    @abstractmethod
    def store_entry(self, record):
        pass
