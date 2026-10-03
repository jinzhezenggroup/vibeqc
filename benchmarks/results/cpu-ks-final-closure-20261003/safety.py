"""wait4 supervisor: true reaped RSS, finite timeout, process-group cleanup."""
import os, signal, subprocess, time, traceback
from pathlib import Path
from common import now, RSS_LIMIT, TIMEOUT

def process_rss(pid):
    try:
        f={s.split(':')[0]:int(s.split()[1])*1024 for s in Path(f'/proc/{pid}/status').read_text().splitlines() if s.startswith(('VmRSS:','VmHWM:'))}
        return f.get('VmRSS',0),f.get('VmHWM',0)
    except (FileNotFoundError,ProcessLookupError):return 0,0

def supervise(command,env,logfile,*,rss_limit=RSS_LIMIT,timeout=TIMEOUT,poll=.05,rss_reader=process_rss):
    start=time.monotonic();p=None;termination=None;signalled=None;peak=0;hwm=0;samples=0
    r=dict(status='starting',started_utc=now(),command=command,rss_limit_bytes=rss_limit,timeout_seconds=timeout,performance_measurement=False)
    def reap(status,usage):
        p.returncode=os.waitstatus_to_exitcode(status)
        r.update(returncode=p.returncode,wait4_status=status,true_ru_maxrss_kib=usage.ru_maxrss,wait4_user_seconds=usage.ru_utime,wait4_system_seconds=usage.ru_stime,wait4_minflt=usage.ru_minflt,wait4_majflt=usage.ru_majflt,wait4_nvcsw=usage.ru_nvcsw,wait4_nivcsw=usage.ru_nivcsw)
    try:
        with Path(logfile).open('x') as log:
            p=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            r['pid']=p.pid
            while True:
                pid,status,usage=os.wait4(p.pid,os.WNOHANG)
                if pid:reap(status,usage);break
                rss,high=rss_reader(p.pid);samples+=1;peak=max(peak,rss);hwm=max(hwm,high)
                if termination is None and max(rss,high)>rss_limit:termination='rss_guard'
                if termination is None and time.monotonic()-start>timeout:termination='timeout'
                if termination and signalled is None:
                    try:os.killpg(p.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    signalled=time.monotonic()
                if signalled is not None and time.monotonic()-signalled>2:
                    try:os.killpg(p.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                time.sleep(poll)
        if r['true_ru_maxrss_kib']*1024>rss_limit:termination=termination or 'rss_guard_postexit'
        if time.monotonic()-start>timeout:termination=termination or 'timeout_postexit'
        r.update(status=termination or ('completed' if p.returncode==0 else 'child_failed'),termination_reason=termination)
    except BaseException:
        error=traceback.format_exc()
        if p is not None and p.returncode is None:
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            try:
                _,status,usage=os.wait4(p.pid,0);reap(status,usage)
            except BaseException:error+='\nREAP ERROR\n'+traceback.format_exc()
        r.update(status='supervisor_failed',termination_reason='supervisor_exception',error=error)
    r.update(finished_utc=now(),elapsed_seconds=time.monotonic()-start,observed_peak_rss_bytes=peak,observed_hwm_bytes=hwm,rss_samples=samples)
    return r
