Task
====

Concurrent task scheduling and execution system for real-time, multi-threaded applications.

Overview
--------

The Task library provides a high-performance framework for concurrent task execution with
dependency management, worker thread pools, and flexible scheduling. It simplifies complex
real-time threading code with a worker queue architecture that is both scalable and deterministic.

Key Features
~~~~~~~~~~~~

-  **Task Graphs**: Define complex task dependencies with automatic execution ordering
-  **Worker Thread Pools**: Configurable worker threads with CPU core pinning and priority
   scheduling
-  **Task Categories**: Organize tasks by category and assign workers to specific categories
-  **Periodic Triggers**: High-precision periodic task execution with nanosecond latencies
-  **Memory Triggers**: Monitor memory locations and execute callbacks on events
-  **Deterministic Execution**: Lock-free and allocation-free critical paths for predictable
   real-time performance
-  **Task Timeouts**: Configurable execution time limits to guarantee real-time operation
-  **Cancellation Support**: Cooperative task cancellation with CancellationToken
-  **Performance Monitoring**: Detailed execution statistics and Chrome trace output

Quick Start
-----------

Creating and Executing a Task
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/task/tests/task_sample_tests.cpp
   :language: cpp
   :start-after: example-begin basic-task-1
   :end-before: example-end basic-task-1
   :dedent: 4

Tasks with Timeout
~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/task/tests/task_sample_tests.cpp
   :language: cpp
   :start-after: example-begin task-timeout-1
   :end-before: example-end task-timeout-1
   :dedent: 4

Tasks with Cancellation
~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../framework/task/tests/task_sample_tests.cpp
   :language: cpp
   :start-after: example-begin task-cancellation-1
   :end-before: example-end task-cancellation-1
   :dedent: 4

Core Concepts
-------------

Task Basics
~~~~~~~~~~~

A **Task** is the fundamental unit of work in the task system. Each task encapsulates:

- A function to execute (lambda or callable)
- Optional timeout duration
- Cancellation token for cooperative cancellation
- Task metadata (name, category, status)
- Dependency relationships with other tasks

Tasks are created using the ``TaskBuilder`` fluent interface and can be executed directly or
scheduled through a TaskScheduler.

Task Graphs
~~~~~~~~~~~

**TaskGraph** provides a fluent API for building complex task dependency graphs. There are two
approaches for creating task graphs:

Single-Task Graphs
^^^^^^^^^^^^^^^^^^

For simple workflows with one task:

.. literalinclude:: ../../../framework/task/tests/task_sample_tests.cpp
   :language: cpp
   :start-after: example-begin simple-graph-1
   :end-before: example-end simple-graph-1
   :dedent: 4

Multi-Task Graphs with Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For complex workflows with multiple tasks and dependencies:

.. literalinclude:: ../../../framework/task/tests/task_sample_tests.cpp
   :language: cpp
   :start-after: example-begin graph-dependencies-1
   :end-before: example-end graph-dependencies-1
   :dedent: 4

Dependencies are expressed by name, and TaskGraph automatically determines execution order
based on dependency chains. Tasks with no unmet dependencies execute immediately, while
dependent tasks wait for their parents to complete.

Task Scheduler
~~~~~~~~~~~~~~

**TaskScheduler** manages worker threads that execute tasks from one or more task graphs.
The scheduler supports:

- Multiple worker threads with configurable counts
- CPU core pinning for deterministic performance
- Real-time thread priorities
- Category-based task routing
- Task readiness checking with configurable tolerance to minimize scheduling jitter

Basic Scheduler Usage
^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/task/tests/task_sample_tests.cpp
   :language: cpp
   :start-after: example-begin basic-scheduler-1
   :end-before: example-end basic-scheduler-1
   :dedent: 4

The scheduler uses a builder pattern for configuration, making it easy to customize behavior
before construction.

Worker Configuration
~~~~~~~~~~~~~~~~~~~~

Workers are the threads that execute tasks. Each worker can be configured with:

- CPU core affinity (pinning to specific cores)
- Thread priority (1-99, higher = more urgent)
- Task categories (which types of tasks this worker handles)

Scheduler with Task Categories
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/task/tests/task_sample_tests.cpp
   :language: cpp
   :start-after: example-begin scheduler-categories-1
   :end-before: example-end scheduler-categories-1
   :dedent: 4

Worker with Core Affinity
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../framework/task/tests/task_sample_tests.cpp
   :language: cpp
   :start-after: example-begin worker-affinity-1
   :end-before: example-end worker-affinity-1
   :dedent: 4

Task Categories
~~~~~~~~~~~~~~~

**TaskCategory** provides an extensible categorization system for organizing tasks. The framework
includes built-in categories (``Default``, ``HighPriority``, ``LowPriority``, ``IO``, ``Compute``,
``Network``, ``Message``), and users can define custom categories.

Custom categories are defined using the ``DECLARE_TASK_CATEGORIES`` macro at namespace scope:

.. literalinclude:: ../../../framework/task/tests/task_sample_tests.cpp
   :language: cpp
   :start-after: example-begin custom-categories-1
   :end-before: example-end custom-categories-1
   :dedent: 0

Once defined, custom categories can be used with tasks and task graphs:

.. literalinclude:: ../../../framework/task/tests/task_sample_tests.cpp
   :language: cpp
   :start-after: example-begin custom-categories-2
   :end-before: example-end custom-categories-2
   :dedent: 4

Categories enable workload partitioning across worker threads, allowing fine-grained control
over task execution resources.

Task Pool
~~~~~~~~~

**TaskPool** provides efficient task object reuse through a lock-free pooling mechanism. Instead
of allocating new tasks for every execution, the pool maintains a cache of reusable task objects.

.. literalinclude:: ../../../framework/task/tests/task_sample_tests.cpp
   :language: cpp
   :start-after: example-begin task-pool-1
   :end-before: example-end task-pool-1
   :dedent: 4

TaskPool is automatically used by TaskGraph for efficient task reuse across multiple scheduling
rounds. The pool tracks statistics including hit rate and reuse counts.

Task Monitor
~~~~~~~~~~~~

**TaskMonitor** provides monitoring of task execution with detailed performance tracking
and visualization capabilities. The monitor operates in a separate background thread and uses
lock-free queues for real-time safe communication with worker threads.

Key capabilities include:

- **Performance Statistics**: Tracks execution duration, scheduling jitter, and task status
- **Timeout Detection**: Automatically detects and can cancel tasks that exceed configured
  timeouts
- **Chrome Trace Output**: Exports execution timelines for visualization in chrome://tracing
- **Graph-Level Analysis**: Provides aggregated statistics grouped by task graph and scheduling
  round

The TaskMonitor is automatically integrated with TaskScheduler. The monitor thread can be
pinned to a specific CPU core for deterministic overhead:

.. literalinclude:: ../../../framework/task/tests/task_sample_tests.cpp
   :language: cpp
   :start-after: example-begin task-monitor-1
   :end-before: example-end task-monitor-1
   :dedent: 4

Triggers
--------

The task library provides two types of triggers for periodic and event-driven task execution.

Timed Trigger
~~~~~~~~~~~~~

**TimedTrigger** executes a callback at regular intervals with nanosecond precision. It supports:

- High-precision periodic execution
- CPU core pinning for deterministic timing
- Real-time thread priorities
- Jump detection for timing anomalies
- Detailed latency statistics

.. literalinclude:: ../../../framework/task/tests/task_sample_tests.cpp
   :language: cpp
   :start-after: example-begin timed-trigger-1
   :end-before: example-end timed-trigger-1
   :dedent: 4

TimedTrigger is commonly used with TaskScheduler to periodically schedule task graphs for
execution, as demonstrated in the sample application.

Memory Trigger
~~~~~~~~~~~~~~

**MemoryTrigger** monitors a memory location and executes a callback when values change.
It supports:

- Atomic memory location monitoring
- Custom comparators for trigger conditions
- Condition variable or polling notification strategies
- CPU core pinning and priority scheduling

.. literalinclude:: ../../../framework/task/tests/task_sample_tests.cpp
   :language: cpp
   :start-after: example-begin memory-trigger-1
   :end-before: example-end memory-trigger-1
   :dedent: 4

MemoryTrigger is useful for event-driven architectures where tasks respond to state changes in
shared memory.

Complete Example
----------------

The task sample application demonstrates a complete workflow:

.. literalinclude:: ../../../framework/task/samples/task_sample.cpp
   :language: cpp
   :start-after: example-begin task-sample-main-1
   :end-before: example-end task-sample-main-1
   :dedent: 8

This example creates a TaskScheduler, defines a simple task graph, and uses TimedTrigger to
periodically schedule the graph for execution at configurable intervals.

Additional Examples
-------------------

For more examples, see:

-  ``framework/task/samples/task_sample.cpp`` - Complete sample application
   with scheduler, triggers, and task graphs
-  ``framework/task/tests/task_sample_tests.cpp`` - Documentation examples and
   unit tests

API Reference
-------------

.. doxygennamespace:: framework::task
   :content-only:
   :members:
   :undoc-members:
