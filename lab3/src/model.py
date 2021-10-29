import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

class QueryArrivedEvent:
    def __init__(self, timepoint):
        self.timepoint = timepoint

    def __str__(self):
        return self.timepoint

class TerminalBreakEvent:
    def __init__(self, timepoint):
        self.timepoint = timepoint

    def __str__(self):
        return self.timepoint

class TerminalRepairEvent:
    def __init__(self, timepoint):
        self.timepoint = timepoint

    def __str__(self):
        return self.timepoint

class QueryProcessedEvent:
    def __init__(self, start_timepoint, timepoint):
        self.start_timepoint = start_timepoint
        self.timepoint = timepoint

    def __str__(self):
        return self.timepoint

class Model:
    def __init__(self, m, lambda_, mu, nu, r, max_queries_processed):
        self._m = m
        self._lambda = lambda_
        self._mu = mu
        self._nu = nu
        self._r = r
        self._max_queries_processed = max_queries_processed

        self._terminal_busy = False
        self._terminal_broken = False
        self._break_timepoint_determined = False
        self._timeline = []
        self._queue = []

        self._queries_processed = 0
        self._queries_dropped = 0

        # states (time recording)
        # firstly with working terminals, next with broken terminals
        self._final_state_durations = [0] * (2 * self._m + 2)
        self._state_durations = []
        self._last_state = 0
        self._last_state_change_timepoint = 0

    def start(self):
        query_gen = self._generate(self._lambda)
        service_gen = self._generate(self._mu)
        break_gen = self._generate(self._nu)
        repair_gen = self._generate(self._r)

        first_query_timepoint = next(query_gen)
        self._timeline.append(QueryArrivedEvent(first_query_timepoint))

        for _ in range(self._max_queries_processed - 1):
            timepoint = self._timeline[len(self._timeline) - 1].timepoint + next(query_gen)
            self._timeline.append(QueryArrivedEvent(timepoint))

        for event in self._timeline:
            if isinstance(event, QueryArrivedEvent):
                if not self._terminal_busy and not self._terminal_broken:
                    self._terminal_busy = True
                    query_processed_timepoint = event.timepoint + next(service_gen)
                    query_processed_event = QueryProcessedEvent(event.timepoint, query_processed_timepoint)
                    self._insert(query_processed_event)

                    if not self._break_timepoint_determined:
                        break_timepoint = event.timepoint + next(break_gen)
                        self._insert(TerminalBreakEvent(break_timepoint))
                        self._break_timepoint_determined = True

                    self._record_state(event.timepoint)
                else:
                    if len(self._queue) < self._m:
                        self._queue.append(event.timepoint)
                        self._record_state(event.timepoint)
                    else:
                        # drop
                        self._queries_dropped += 1
        
            if isinstance(event, QueryProcessedEvent):
                self._queries_processed += 1
                self._terminal_busy = False

                if len(self._queue) != 0:
                    waiting_start = self._queue.pop(0)
                    self._terminal_busy = True
                    query_processed_timepoint = event.timepoint + next(service_gen)
                    query_processed_event = QueryProcessedEvent(waiting_start, query_processed_timepoint)
                    self._insert(query_processed_event)

                    if not self._break_timepoint_determined:
                        break_timepoint = event.timepoint + next(break_gen)
                        self._insert(TerminalBreakEvent(break_timepoint))
                        self._break_timepoint_determined = True

                self._record_state(event.timepoint)

            if isinstance(event, TerminalBreakEvent) and event != self._timeline[-1]:
                self._break_timepoint_determined = False
                self._terminal_broken = True

                removed_event = self._remove_nearest_query_processed_event(event)

                repair_timepoint = event.timepoint + next(repair_gen)
                repair_event = TerminalRepairEvent(repair_timepoint)
                self._insert(repair_event)

                if len(self._queue) < self._m:
                    self._queue.append(removed_event.start_timepoint if removed_event != None else event.timepoint)
                else:
                    # drop
                    self._queries_dropped += 1

                self._record_state(event.timepoint)

            if isinstance(event, TerminalRepairEvent):
                self._terminal_broken = False
                self._terminal_busy = False

                if len(self._queue) != 0:
                    waiting_start = self._queue.pop(0)
                    self._terminal_busy = True
                    query_processed_timepoint = event.timepoint + next(service_gen)
                    query_processed_event = QueryProcessedEvent(waiting_start, query_processed_timepoint)
                    self._insert(query_processed_event)

                    if not self._break_timepoint_determined:
                        break_timepoint = event.timepoint + next(break_gen)
                        self._insert(TerminalBreakEvent(break_timepoint))
                        self._break_timepoint_determined = True

                self._record_state(event.timepoint)
                
                    
    def _generate(self, param):
        while True:
            yield np.random.exponential(1 / param)

    def _insert(self, event):
        for i in range(1, len(self._timeline)):
            if event.timepoint > self._timeline[i - 1].timepoint and event.timepoint < self._timeline[i].timepoint:
                self._timeline.insert(i, event)
                break
        else:
            self._timeline.append(event)

    def _remove_nearest_query_processed_event(self, terminal_break_event):
        break_event_index = self._timeline.index(terminal_break_event)

        for i in range(break_event_index, len(self._timeline)):
            if isinstance(self._timeline[i], QueryProcessedEvent):
                return self._timeline.pop(i)
            

    def _record_state(self, timepoint):
        time_delta = timepoint - self._last_state_change_timepoint
        self._final_state_durations[self._last_state] += time_delta
        
        self._last_state_change_timepoint = timepoint

        if not self._terminal_broken:
            self._last_state = int(self._terminal_busy) + len(self._queue)
        else:
            self._last_state = 1 + self._m + len(self._queue)

        self._state_durations.append((timepoint, self._final_state_durations.copy()))
    
    def show_stationary_stats(self):
        x = [timepoint for timepoint, _ in self._state_durations]

        for i in range(len(self._final_state_durations)):
            y = [durations[i] / timepoint for timepoint, durations in self._state_durations]
            plt.plot(x, y)

        plt.show()

    def show_stats(self):
        print('Queries processed:', self._queries_processed)
        print('Quries dropped:', self._queries_dropped)

        return [duration / self._timeline[-1].timepoint for duration in self._final_state_durations]

print('THEORETICAL STATS: ')

A = np.array([
    [-2, 1, 0, 0, 0, 0],
    [2, -3.5, 1, 0, 1, 0],
    [0, 2, -3.5, 1, 0, 1],
    [0, 0, 2, -1.5, 0, 0],
    [0, 0.5, 0, 0, -3, 0],
    [0, 0, 0.5, 0.5, 2, -1],
    [1, 1, 1, 1, 1, 1]
])

b = np.array([0, 0, 0, 0, 0, 0, 1])

def f(x):
    y = np.dot(A, x) - b
    return np.dot(y, y)

cons = (
    {'type': 'ineq', 'fun': lambda x: x[0]},
    {'type': 'ineq', 'fun': lambda x: x[1]},
    {'type': 'ineq', 'fun': lambda x: x[2]},
    {'type': 'ineq', 'fun': lambda x: x[3]},
    {'type': 'ineq', 'fun': lambda x: x[4]},
    {'type': 'ineq', 'fun': lambda x: x[5]},
)
t_probs = opt.minimize(f, [0, 0, 0, 0, 0, 0], method='SLSQP', constraints=cons, options={'disp': False}).x

p1 = t_probs[0]
p2 = t_probs[1]
p3 = t_probs[2]
p4 = t_probs[3]
p5 = t_probs[4]
p6 = t_probs[5]

print('terminal is free, 0 in queue | p = ', p1)
print('terminal is busy, 0 in queue | p = ', p2)
print('terminal is busy, 1 in queue | p = ', p3)
print('terminal is busy, 2 in queue | p = ', p4)
print('terminal is broken, 1 in queue | p = ', p5)
print('terminal is broken, 2 in queue | p = ', p6)

t_p_denial = p4 + p6
t_avg_queue_length = 1 * (p3 + p5) + 2 * (p4 + p6)
t_avg_processing = p2 + p3 + p4
t_Q = 1 - t_p_denial
t_A = 2 * t_Q
t_avg_waiting_time = t_avg_queue_length / 2
t_avg_processing_time = t_Q / 1
t_avg_total_time = t_avg_processing_time + t_avg_waiting_time
print('Probability of denial: ', t_p_denial)
print('Average queue length: ', t_avg_queue_length)
print('Average processing: ', t_avg_processing)
print('Relative throughput: ', t_Q)
print('Absolute throрugput', t_A)
print('Average processing time: ', t_avg_processing_time)
print('Average waiting time: ', t_avg_waiting_time)
print('Average time in system: ', t_avg_total_time)

print('_______________________________')
print('EMPIRICAL STATS: ')

n = 1
m = 2
lambda_ = 2
mu = 1
nu = 0.5
r = 1
model = Model(m, lambda_, mu, nu, r, 1000)
model.start()
e_probs = model.show_stats()
p1 = e_probs[0]
p2 = e_probs[1]
p3 = e_probs[2]
p4 = e_probs[3]
p5 = e_probs[4]
p6 = e_probs[5]

print('terminal is free, 0 in queue | p = ', p1)
print('terminal is busy, 0 in queue | p = ', p2)
print('terminal is busy, 1 in queue | p = ', p3)
print('terminal is busy, 2 in queue | p = ', p4)
print('terminal is broken, 1 in queue | p = ', p5)
print('terminal is broken, 2 in queue | p = ', p6)

e_p_denial = p4 + p6
e_avg_queue_length = 1 * (p3 + p5) + 2 * (p4 + p6)
e_avg_processing = p2 + p3 + p4
e_Q = 1 - e_p_denial
e_A = 2 * e_Q
e_avg_waiting_time = e_avg_queue_length / 2
e_avg_processing_time = e_Q / 1
e_avg_total_time = e_avg_processing_time + e_avg_waiting_time
print('Probability of denial: ', e_p_denial)
print('Average queue length: ', e_avg_queue_length)
print('Average processing: ', e_avg_processing)
print('Relative throughput: ', e_Q)
print('Absolute throрugput', e_A)
print('Average processing time: ', e_avg_processing_time)
print('Average waiting time: ', e_avg_waiting_time)
print('Average time in system: ', e_avg_total_time)

EPS = 0.1

assert abs(e_p_denial - t_p_denial)
assert abs(e_Q - t_Q) < EPS
assert abs(e_A - t_A) < EPS
assert abs(e_avg_processing - t_avg_processing) < EPS
assert abs(e_avg_queue_length - t_avg_queue_length) < EPS
assert abs(e_avg_processing_time - t_avg_processing_time) < EPS
assert abs(e_avg_waiting_time - t_avg_waiting_time) < EPS
assert abs(e_avg_total_time - t_avg_total_time) < EPS

print('Model is working correctly')

_, ax = plt.subplots(1, 2)
ax[0].title.set_text(f'Empirical probabilities (n={n}, m={m}, lambda={lambda_}, mu={mu}, nu={nu}, r={r})')
ax[0].hist(list(np.arange(0, len(e_probs), 1)), weights=e_probs)
ax[1].title.set_text(f'Theoretical probabilities (n={n}, m={m}, lambda={lambda_}, mu={mu}, nu={nu}, r={r})')
ax[1].hist(list(np.arange(0, len(t_probs), 1)), weights=t_probs)
plt.show()

model.show_stationary_stats()