"""Managed worker lifecycle using a virtual clock, inert subprocesses, and no MLX."""
import asyncio
import importlib.util
import sys
from types import SimpleNamespace

import pytest


class Worker:
    def __init__(self, pid, started, clock, kind, stderr):
        self.pid = pid
        self.started = started
        self.clock = clock
        self.kind = kind
        self.stderr = stderr
        self.returncode = None

    def poll(self):
        if self.clock() - self.started >= 8 and self.kind in ('marked', 'exit_only', 'ordinary'):
            if self.returncode is None and self.kind != 'exit_only':
                self.stderr.write('SABLE_MTP_VERIFY_PROBE_TIMEOUT\n')
                self.stderr.flush()
            self.returncode = 75 if self.kind == 'ordinary' else 76
        return self.returncode

    def healthy(self):
        return self.kind == 'slow_ready' and self.clock() - self.started >= 8


@pytest.fixture
def manager(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'mlx', None)
    monkeypatch.setitem(sys.modules, 'mlx.core', None)
    from mlx_serve import process_manager
    from mlx_serve.config import ModelConfig

    # Independent module globals retain real spawn, wait, health, and status code.
    name = 'mlx_serve._sable_process_manager_for_test'
    spec = importlib.util.spec_from_file_location(name, process_manager.__file__)
    pm = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, pm)
    spec.loader.exec_module(pm)
    config = ModelConfig(name='qwen-test', type='vision', hf_path='acme/tiny', draft_kind='mtp', mtp_verify_scan='joint_v1')
    scenario = SimpleNamespace(now=100.0, kinds=[], workers=[], envs=[], events=[], wait_budgets=[])
    clock = lambda: scenario.now
    pm.time = SimpleNamespace(monotonic=clock, time=clock)
    pm._LOG_DIR = tmp_path / 'logs'
    monkeypatch.setenv('SABLE_MTP_VERIFY_PREFERENCE', 'joint-first')
    monkeypatch.setattr(pm.config, 'MODELS', {config.name: config})
    monkeypatch.setattr(pm.config, 'STARTUP_TIMEOUT', 10)
    monkeypatch.setattr(pm.events, 'emit', lambda event, **fields: scenario.events.append((event, fields)))
    snapshot = SimpleNamespace(ram_used_gb=0.25, subprocess_rss_mb=5, metal_active_mb=None, ram_available_gb=8)
    monkeypatch.setattr(pm.metrics, 'take_memory_snapshot', lambda **kw: snapshot)
    pm.psutil = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(percent=10))
    pm._is_model_cached = lambda cfg: True
    pm._build_command = lambda cfg: ['inert-worker']

    def spawn(cmd, *, stderr, env, **kwargs):
        assert cmd == ['inert-worker']
        worker = Worker(len(scenario.workers) + 100, clock(), clock, scenario.kinds.pop(0), stderr)
        scenario.workers.append(worker)
        scenario.envs.append(env)
        return worker

    async def sleep(seconds):
        scenario.now += seconds
        await asyncio.sleep(0)

    async def wait_for(awaitable, timeout):
        scenario.wait_budgets.append(timeout)
        deadline = clock() + timeout
        task = asyncio.ensure_future(awaitable)
        while not task.done():
            await asyncio.sleep(0)
            if clock() > deadline and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise TimeoutError
        return task.result()

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def get(self, url, timeout):
            assert url.endswith('/health')
            return SimpleNamespace(status_code=200 if scenario.workers[-1].healthy() else 503)

    pm.asyncio = SimpleNamespace(Event=asyncio.Event, create_task=asyncio.create_task, sleep=sleep, wait_for=wait_for)
    pm.subprocess = SimpleNamespace(Popen=spawn, DEVNULL=-3)
    pm.httpx = SimpleNamespace(AsyncClient=Client, RequestError=RuntimeError)
    yield pm, scenario
    if pm._stderr_log_handle is not None:
        pm._stderr_log_handle.close()


def test_late_optional_exit_and_slow_reference_share_one_bounded_request(manager):
    pm, scenario = manager
    scenario.kinds = ['marked', 'slow_ready']
    assert asyncio.run(pm.ensure_model('qwen-test')) is True
    assert scenario.now == 116
    assert len(scenario.workers) == 2
    assert scenario.wait_budgets and all(20 <= value <= 25 for value in scenario.wait_budgets)
    assert [e.get('SABLE_MTP_VERIFY_RETRY') for e in scenario.envs] == [None, 'probe_timeout']
    assert pm.get_status(reap=False)['attempt_state'] == 'ready'


@pytest.mark.parametrize(('kinds', 'elapsed', 'reason'), [
    (['marked', 'marked'], 16, 'process_crash'),
    (['exit_only'], 8, 'process_crash'),
    (['ordinary'], 8, 'process_crash'),
    (['marked', 'never_ready'], 18, 'startup_timeout'),
])
def test_uncorroborated_or_repeated_exit_and_slow_retry_fail(manager, kinds, elapsed, reason):
    from fastapi import HTTPException

    pm, scenario = manager
    scenario.kinds = list(kinds)
    with pytest.raises(HTTPException) as failure:
        asyncio.run(pm.ensure_model('qwen-test'))
    assert failure.value.status_code == 503
    assert len(scenario.workers) == len(kinds)
    assert scenario.now == 100 + elapsed
    assert pm.get_status(reap=False)['failure_reason'] == reason


def test_dead_ready_worker_is_observed_without_reaping_and_history_survives_reload(manager):
    pm, scenario = manager
    scenario.kinds = ['slow_ready', 'slow_ready']
    assert asyncio.run(pm.ensure_model('qwen-test'))
    first = pm.get_status(reap=False)
    worker = pm._process
    worker.returncode = -9
    events = list(scenario.events)
    failure = pm.get_status(reap=False)
    assert failure['state'] == failure['attempt_state'] == 'failed'
    assert failure['failure_reason'] == 'worker_exit'
    assert failure['pid'] is None and failure['attempt_id'] == first['attempt_id']
    assert pm._process is worker and pm._state is pm.ModelState.READY
    assert scenario.events == events and len(scenario.workers) == 1

    assert asyncio.run(pm.ensure_model('qwen-test'))
    recovered = pm.get_status(reap=False)
    assert recovered['state'] == recovered['attempt_state'] == 'ready'
    assert recovered['attempt_id'] != first['attempt_id'] and recovered['failure_reason'] is None
    history = recovered['last_failed_attempt']
    assert history['attempt_id'] == first['attempt_id'] and history['reason'] == 'worker_exit'
    assert history['model'] == 'qwen-test' and history['failure_at'] <= scenario.now
