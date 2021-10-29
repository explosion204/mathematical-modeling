import numpy as np
from mpmath import nsum, nprod, fac
from statistics import mean
import matplotlib.pyplot as plt

class QueryArrivedEvent:
    def __init__(self, timepoint):
        self.timepoint = timepoint

    def __str__(self):
        return self.timepoint

class QueryProcessedEvent:
    def __init__(self, start_timepoint, timepoint, terminal_no):
        self.start_timepoint = start_timepoint
        self.timepoint = timepoint
        self.terminal_no = terminal_no

    def __str__(self):
        return self.timepoint

class Model:
    def __init__(self, n, m, lambda_, mu, nu, max_queries_processed, eps):
        self._n = n
        self._m = m
        self._lambda = lambda_
        self._mu = mu
        self._nu = nu
        self._max_queries_processed = max_queries_processed
        self._eps = eps
        
        self._terminal_availabilities = [True] * n
        self._timeline = []
        self._queue = []

        self._busy_terminals = 0
        self._state_durations = []
        self._final_state_durations = [0] * (n + m + 1)
        self._last_state = 0
        self._last_state_change_timepoint = 0

        self._queries_processed = 0
        self._queries_dropped = 0

        self._processing_time = []
        self._waiting_time = []


    def start(self):
        query_gen = self._generate(self._lambda)
        service_gen = self._generate(self._mu)
        waiting_gen = self._generate(self._nu)

        first_timepoint = next(query_gen)
        self._timeline.append(QueryArrivedEvent(first_timepoint))

        for _ in range(self._max_queries_processed - 1):
            timepoint = self._timeline[len(self._timeline) - 1].timepoint + next(query_gen)
            self._timeline.append(QueryArrivedEvent(timepoint))

        for event in self._timeline:
            self._clean_queue(event.timepoint)

            if isinstance(event, QueryArrivedEvent):
                has_available_terminals, terminal_no = self._find_terminal()

                if has_available_terminals:
                    query_processed_timepoint = event.timepoint + next(service_gen)
                    query_processed_event = QueryProcessedEvent(event.timepoint, query_processed_timepoint, terminal_no)
                    self._insert(query_processed_event)
                    self._record_state(event.timepoint)
                    self._waiting_time.append(0)
                else:
                    if len(self._queue) < self._m:
                        waiting_deadline = event.timepoint + next(waiting_gen)
                        self._queue.append((event.timepoint, waiting_deadline))
                        self._record_state(event.timepoint)
                    else:
                        # drop
                        self._queries_dropped += 1

            if isinstance(event, QueryProcessedEvent):
                self._queries_processed += 1
                self._busy_terminals -= 1
                self._terminal_availabilities[event.terminal_no] = True
                self._processing_time.append(event.timepoint - event.start_timepoint)

                if len(self._queue) != 0:
                    start_timepoint, _ = self._queue.pop(0)

                    query_processed_timepoint = event.timepoint + next(service_gen)
                    query_processed_event = QueryProcessedEvent(event.timepoint, query_processed_timepoint, event.terminal_no)
                    self._terminal_availabilities[event.terminal_no] = False
                    self._busy_terminals += 1
                    self._insert(query_processed_event)
                    self._waiting_time.append(event.timepoint - start_timepoint)

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

    def _find_terminal(self):
        for i, status in enumerate(self._terminal_availabilities):
            if status:
                self._busy_terminals += 1
                self._terminal_availabilities[i] = False

                return True, i
        
        return False, -1

    def _clean_queue(self, current_timepoint):
        cleaned_queue = []

        while len(self._queue) != 0:
            start_timepoint, waiting_deadline = self._queue.pop(0)

            if current_timepoint > waiting_deadline:
                # drop
                self._queries_dropped += 1
                self._record_state(waiting_deadline)

            else:
                cleaned_queue.append((start_timepoint, waiting_deadline))

        self._queue = cleaned_queue
                

    def _record_state(self, timepoint):
        time_delta = timepoint - self._last_state_change_timepoint
        self._final_state_durations[self._last_state] += time_delta

        self._last_state_change_timepoint = timepoint
        self._last_state = self._busy_terminals + len(self._queue)

        self._state_durations.append((timepoint, self._final_state_durations.copy()))

    def _show_empirical_stats(self):
        print('EMPIRICAL STATS')
        print('Queries processed:', self._queries_processed)
        print('Quries dropped:', self._queries_dropped)

        avg_processing = 0
        probs = []
        for k in range(self._n + 1):
            pk = self._final_state_durations[k] / self._timeline[-1].timepoint
            probs.append(pk)
            avg_processing += k * pk
            print(f'{k} terminals are busy & 0 queries in queue | p{k} =', pk)

        avg_queue_length = 0
        for s in range(self._n + 1, self._n + self._m + 1):
            ps = self._final_state_durations[s] / self._timeline[-1].timepoint
            probs.append(ps)
            avg_queue_length += (s - self._n) * ps
            print(f'{self._n} terminals are busy & {s - self._n} queries in queue | pn+{s - self._n} =', ps)

        p_denial = ps
        Q = 1 - p_denial
        A = self._lambda * Q
        avg_waiting_time = mean(self._waiting_time)
        avg_processing_time = mean(self._processing_time)
        avg_total_time = avg_waiting_time + avg_processing_time
        print('Probability of denial: ', p_denial)
        print('Relative throughput: ', Q)
        print('Absolute throрugput', A)
        print('Average queue length: ', avg_queue_length)
        print('Average processing queries: ', avg_processing)
        print('Average queries in system: ', avg_processing + avg_queue_length)
        print('Average processing time: ', avg_processing_time)
        print('Average waiting time: ', avg_waiting_time)
        print('Average time in system: ', avg_total_time)

        return probs, p_denial, Q, A, avg_processing, avg_queue_length, avg_processing + avg_queue_length, avg_processing_time, avg_waiting_time, avg_total_time

    def _show_theoretical_stats(self):
        print('THEORETICAL STATS')
        ro = self._lambda / self._mu
        p0 = 1 / (nsum(lambda k: ro ** k / fac(k), [0, self._n]) + (ro ** self._n / fac(self._n)) * nsum(lambda i: ro ** i / nprod(lambda l: (self._n + l * self._nu / self._mu), [1, i]), [1, self._m])) 
        print('0 terminals are busy & 0 queries in queue | p0 =', p0)

        avg_processing = 0
        probs = [p0]
        for k in range(1, self._n + 1):
            pk = ro ** k * p0 / fac(k)
            probs.append(pk)
            avg_processing += k * pk
            print(f'{k} terminals are busy & 0 queries in queue | p{k} =', pk)

        avg_queue_length = 0
        for s in range(1, self._m + 1):
            ps = ro ** (self._n + s) * p0 / (fac(self._n) * nprod(lambda l: self._n + l * self._nu / self._mu, [1, s]))
            probs.append(ps)
            avg_queue_length += s * ps
            print(f'{self._n} terminals are busy & {s} queries in queue | pn+{s} =', ps)

        p_denial = ps
        Q = 1 - p_denial
        A = self._lambda * Q
        avg_waiting_time = avg_queue_length / self._lambda
        avg_processing_time = Q / self._mu
        avg_total_time = avg_processing_time + avg_waiting_time
        print('Probability of denial: ', p_denial)
        print('Relative throughput: ', Q)
        print('Absolute throрugput', A)
        print('Average queue length: ', avg_queue_length)
        print('Average processing queries: ', avg_processing)
        print('Average queries in system: ', avg_processing + avg_queue_length)
        print('Average processing time: ', avg_processing_time)
        print('Average waiting time: ', avg_waiting_time)
        print('Average time in system: ', avg_total_time)

        return probs, p_denial, Q, A, avg_processing, avg_queue_length, avg_processing + avg_queue_length, avg_processing_time, avg_waiting_time, avg_total_time

    def _show_stationary_stats(self):
        x = [timepoint for timepoint, _ in self._state_durations]

        for i in range(len(self._final_state_durations)):
            y = [durations[i] / timepoint for timepoint, durations in self._state_durations]
            plt.plot(x, y)

        plt.show()

    def show_stats(self):
        print('______________________________________________')
        print(f'PARAMS: n={self._n}, m={self._m}, lambda={self._lambda}, mu={self._mu}, nu={self._nu}')
        e_probs, e_p_denial, e_Q, e_A, e_avg_processing, e_avg_queue_length, e_avg_total, e_avg_processing_time, e_avg_waiting_time, e_avg_total_time = model._show_empirical_stats()
        print()
        t_probs, t_p_denial, t_Q, t_A, t_avg_processing, t_avg_queue_length, t_avg_total, t_avg_processing_time, t_avg_waiting_time, t_avg_total_time = model._show_theoretical_stats()

        assert abs(e_p_denial - t_p_denial)
        assert abs(e_Q - t_Q) < self._eps
        assert abs(e_A - t_A) < self._eps
        assert abs(e_avg_processing - t_avg_processing) < self._eps
        assert abs(e_avg_queue_length - t_avg_queue_length) < self._eps
        assert abs(e_avg_total - t_avg_total) < self._eps
        assert abs(e_avg_processing_time - t_avg_processing_time) < self._eps
        assert abs(e_avg_waiting_time - t_avg_waiting_time) < self._eps
        assert abs(e_avg_total_time - t_avg_total_time) < self._eps
        print('MODEL IS CORRECT')

        _, ax = plt.subplots(1, 2)
        ax[0].title.set_text(f'Empirical probabilities (n={self._n}, m={self._m}, lambda={self._lambda}, mu={self._mu}, nu={self._nu})')
        ax[0].hist(list(np.arange(0, len(e_probs), 1)), weights=e_probs)
        ax[1].title.set_text(f'Theoretical probabilities (n={self._n}, m={self._m}, lambda={self._lambda}, mu={self._mu}, nu={self._nu})')
        ax[1].hist(list(np.arange(0, len(t_probs), 1)), weights=t_probs)
        plt.show()

        model._show_stationary_stats()

        print('______________________________________________')
        

params_set = [
    (2, 3, 2, 3, 1, 1000, 0.05),
    (3, 4, 3, 3, 2, 1000, 0.05),
    (5, 5, 1, 1, 1, 1000, 0.05)
]

for params in params_set:
    model = Model(*params)
    model.start()
    model.show_stats()
