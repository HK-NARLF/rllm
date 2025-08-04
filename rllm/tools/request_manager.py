import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any, Coroutine
from threading import Thread, Lock

from rllm.tools.tool_base import ToolOutput
from rllm.tools.registry import ToolRegistry

# A request tuple that will be placed in the queue
ToolRequest = namedtuple("ToolRequest", ["priority", "timestamp", "request_id", "tool_name", "tool_args", "original_call_id"])


class BaseScheduler(ABC):
    """
    Abstract base class for a request scheduler.
    Allows for custom logic on how to enqueue requests, e.g., simple FIFO,
    rate-limited, or priority-based scheduling.
    """

    @abstractmethod
    async def enqueue(self, queue: asyncio.PriorityQueue, request: ToolRequest):
        """
        Adds a request to the queue according to the scheduling strategy.

        Args:
            queue: The asyncio.PriorityQueue to add the request to.
            request: The ToolRequest object.
        """
        raise NotImplementedError


class FIFOScheduler(BaseScheduler):
    """
    A simple First-In, First-Out scheduler.
    It uses the request's timestamp as its priority, ensuring older requests
    are processed first.
    """

    async def enqueue(self, queue: asyncio.PriorityQueue, request: ToolRequest):
        # Priority is the timestamp, so older requests go first.
        await queue.put(request)


class ToolRequestManager:
    """
    A centralized, singleton manager for handling all tool requests.

    This manager uses a worker pool to process tool calls from a priority queue,
    allowing for centralized control, scheduling, and rate-limiting.
    """
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.queue: asyncio.PriorityQueue[ToolRequest] = asyncio.PriorityQueue()
        self.results: dict[str, Any] = {}
        self.workers: list[asyncio.Task] = []
        self.scheduler: BaseScheduler = FIFOScheduler()
        self._is_running = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: Thread | None = None
        self._lock = Lock()
        self._initialized = True

    def _start_event_loop(self):
        """Runs the event loop in a separate thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _worker(self):
        """The worker coroutine that pulls requests from the queue and executes them."""
        while self._is_running:
            try:
                request: ToolRequest = await self.queue.get()
                request_id = request.request_id

                # Using a future from the correct loop to signal completion
                result_future = self.results.get(request_id)
                if not result_future:
                    self.queue.task_done()
                    continue

                try:
                    # Get the tool registry instance
                    tool_registry = ToolRegistry()
                    tool_cls = tool_registry.get(request.tool_name)
                    if tool_cls is None:
                        raise ValueError(f"Tool '{request.tool_name}' not found in registry.")

                    tool_instance = tool_cls()
                    
                    if asyncio.iscoroutinefunction(tool_instance.async_forward):
                        result: ToolOutput = await tool_instance.async_forward(**request.tool_args)
                    else:
                        result = await self._loop.run_in_executor(None, tool_instance.forward, **request.tool_args)

                    result_future.set_result(result)

                except Exception as e:
                    result_future.set_exception(e)
                finally:
                    self.queue.task_done()

            except asyncio.CancelledError:
                break
    
    def start(self, num_workers: int = 4, scheduler: BaseScheduler | None = None):
        with self._lock:
            if self._is_running:
                return
                
            if scheduler:
                self.scheduler = scheduler

            self._is_running = True
            self._thread = Thread(target=self._start_event_loop, daemon=True)
            self._thread.start()

            # Wait briefly for the loop to start
            while self._loop is None:
                time.sleep(0.01)

            for _ in range(num_workers):
                self.workers.append(asyncio.run_coroutine_threadsafe(self._worker(), self._loop))
            
            print(f"ToolRequestManager started with {num_workers} workers and {self.scheduler.__class__.__name__}.")

    def stop(self):
        with self._lock:
            if not self._is_running or not self._loop:
                return
            
            async def _shutdown():
                for worker_future in self.workers:
                    worker_future.cancel()
                await asyncio.gather(*[asyncio.wrap_future(wf) for wf in self.workers], return_exceptions=True)
                self._loop.stop()

            asyncio.run_coroutine_threadsafe(_shutdown(), self._loop)
            self._thread.join()
            self._is_running = False
            print("ToolRequestManager stopped.")

    def submit_request(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        original_call_id: str,
        priority: int = 10
    ) -> Any:
        if not self._is_running or not self._loop:
            raise RuntimeError("Manager is not running. Call start() first.")

        request_id = str(uuid.uuid4())
        
        # Create a future in the manager's event loop
        future_future = asyncio.run_coroutine_threadsafe(self._create_and_submit_request(request_id, tool_name, tool_args, original_call_id, priority), self._loop)
        
        # Return a future that will be completed when the request is processed
        return future_future

    async def _create_and_submit_request(self, request_id: str, tool_name: str, tool_args: dict[str, Any], original_call_id: str, priority: int):
        """Creates a future in the manager's event loop and submits the request."""
        # Create the future in the manager's event loop
        future = asyncio.Future()
        self.results[request_id] = future
        
        request = ToolRequest(
            priority=priority,
            timestamp=time.time(),
            request_id=request_id,
            tool_name=tool_name,
            tool_args=tool_args,
            original_call_id=original_call_id
        )

        # Enqueue the request using the scheduler
        await self.scheduler.enqueue(self.queue, request)
        
        return future 